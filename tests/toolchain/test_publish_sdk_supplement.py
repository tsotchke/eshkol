import hashlib
import base64
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2] if '.scratch' not in str(Path(__file__).parent) else Path.cwd()
SPEC = importlib.util.spec_from_file_location('sdk_publisher', ROOT / 'scripts/publish_sdk_supplement.py')
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)
ICC_DYNAMIC_IMPORT_TARGETS = ['publish_sdk_supplement']


class FakeGitHub:
    def __init__(self, source):
        self.source = source
        self.uploads = 0
        self.payload = None
        self.release = {'id': 7, 'tag_name': 'v1.3.5-evolve', 'draft': False, 'assets': []}

    def json(self, endpoint):
        if '/contents/LICENSE' in endpoint:
            return {'encoding': 'base64', 'content': base64.b64encode(b'fixture license').decode()}
        if '/git/ref/' in endpoint:
            return {'object': {'type': 'tag', 'sha': 'b' * 40}}
        if '/git/tags/' in endpoint:
            return {'object': {'type': 'commit', 'sha': self.source}}
        return self.release

    def upload(self, tag, archive):
        self.uploads += 1
        self.payload = archive.read_bytes()
        self.release['assets'] = [{'id': 9, 'name': archive.name, 'state': 'uploaded',
                                  'size': len(self.payload), 'digest': 'sha256:' + hashlib.sha256(self.payload).hexdigest(),
                                  'uploader': {'login': 'fixture-publisher', 'id': 12},
                                  'browser_download_url': 'https://github.com/tsotchke/eshkol/releases/download/' + tag + '/' + archive.name}]

    def download(self, asset_id, destination):
        destination.write_bytes(self.payload)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / '.scratch'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.source = 'a' * 40
        self.compiler = bytes.fromhex('cffaedfe0c000001') + b'fixture compiler'
        self.manifest = {'schema': 'tsotchke.eshkol.release.v1', 'version': '1.3.5-evolve',
                         'source_commit': self.source, 'build_id': 'c' * 64, 'publisher_verified': False,
                         'artifacts': {'compiler': {'path': 'bin/eshkol-run', 'sha256': hashlib.sha256(self.compiler).hexdigest()}}}
        self.packet = {'platform': 'darwin-arm64', 'source_commit': self.source, 'candidate_build_id': 'c' * 64}
        self.make_bundle()
        self.api = FakeGitHub(self.source)

    def make_bundle(self, extra=None, compiler=None):
        archive = self.work / 'sdk.tar.gz'
        raw = json.dumps(self.manifest).encode()
        entries = {'manifest.candidate.json': raw, 'LICENSE': b'fixture license', 'bin/eshkol-run': compiler or self.compiler}
        if extra:
            entries.update(extra)
        with tarfile.open(archive, 'w:gz') as bundle:
            for name, data in entries.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
        self.packet.update({'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                            'archive': {'path': str(archive), 'sha256': publisher.digest_file(archive), 'size': archive.stat().st_size}})
        self.packet_path = self.work / 'packet.json'
        self.packet_path.write_text(json.dumps(self.packet))

    def test_prepare_validates_inventory_and_does_not_upload(self):
        plan = publisher.prepare(self.packet_path, self.api)
        self.assertFalse(plan['publisher_verified'])
        self.assertFalse(plan['production_qualified'])
        self.assertEqual(self.api.uploads, 0)

    def test_prepare_reuses_existing_content_under_another_name(self):
        self.api.release['assets'] = [{'name': 'existing-sdk.tar.gz',
                                      'size': self.packet['archive']['size'],
                                      'digest': 'sha256:' + self.packet['archive']['sha256']}]
        plan = publisher.prepare(self.packet_path, self.api)
        self.assertEqual(plan['asset_name'], 'existing-sdk.tar.gz')
        self.assertEqual(self.api.uploads, 0)

    def test_archive_tamper_rejected(self):
        Path(self.packet['archive']['path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            publisher.prepare(self.packet_path, self.api)

    def test_unsafe_member_rejected(self):
        self.make_bundle(extra={'../escape': b'bad'})
        with self.assertRaises(ValueError):
            publisher.prepare(self.packet_path, self.api)

    def test_artifact_content_tamper_rejected(self):
        self.make_bundle(compiler=b'changed compiler')
        with self.assertRaises(ValueError):
            publisher.prepare(self.packet_path, self.api)

    def test_wrong_source_tag_rejected(self):
        self.api.source = 'd' * 40
        with self.assertRaises(ValueError):
            publisher.prepare(self.packet_path, self.api)

    def test_publish_requires_reviewed_digest(self):
        plan = publisher.prepare(self.packet_path, self.api)
        with self.assertRaises(ValueError):
            publisher.publish(plan, self.api, self.work / 'upload', 'd' * 64)
        self.assertEqual(self.api.uploads, 0)

    def test_upload_retrieval_receipt_and_matching_retry(self):
        plan = publisher.prepare(self.packet_path, self.api)
        receipt = publisher.publish(plan, self.api, self.work / 'upload', plan['archive_sha256'])
        self.assertTrue(receipt['publisher_verified'])
        self.assertFalse(receipt['production_qualified'])
        self.assertEqual(self.api.uploads, 1)
        publisher.publish(plan, self.api, self.work / 'retry', plan['archive_sha256'])
        self.assertEqual(self.api.uploads, 1)

    def test_conflicting_existing_asset_never_replaced(self):
        plan = publisher.prepare(self.packet_path, self.api)
        self.api.release['assets'] = [{'name': plan['asset_name'], 'id': 9, 'state': 'uploaded',
                                      'size': 1, 'digest': 'sha256:' + 'd' * 64}]
        with self.assertRaises(ValueError):
            publisher.publish(plan, self.api, self.work / 'conflict', plan['archive_sha256'])
        self.assertEqual(self.api.uploads, 0)

    def test_download_mismatch_cannot_emit_verified_receipt(self):
        plan = publisher.prepare(self.packet_path, self.api)
        self.api.download = lambda _id, path: path.write_bytes(b'wrong download')
        with self.assertRaises(ValueError):
            publisher.publish(plan, self.api, self.work / 'download', plan['archive_sha256'])
        self.assertFalse((self.work / 'download/publisher-receipt.json').exists())


if __name__ == '__main__':
    unittest.main()
