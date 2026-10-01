#!/usr/bin/env python3
"""Validate and publish an immutable SDK supplement to an existing release."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile

REPOSITORY = "tsotchke/eshkol"
HEX = re.compile(r"[0-9a-f]{64}\Z")
OID = re.compile(r"[0-9a-f]{40}\Z")
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9-]+)?\Z")
MAX_BYTES = 512 * 2**20


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_packet(packet_path):
    packet = json.loads(Path(packet_path).read_text())
    archive = Path(packet["archive"]["path"]).resolve()
    expected = packet["archive"]["sha256"]
    if not HEX.fullmatch(expected) or digest_file(archive) != expected:
        raise ValueError("SDK archive digest differs from reviewed packet")
    if archive.stat().st_size != packet["archive"]["size"]:
        raise ValueError("SDK archive size differs from reviewed packet")
    if packet["platform"] != "darwin-arm64":
        raise ValueError("this publisher recipe qualifies darwin-arm64 only")
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) > 2000 or sum(m.size for m in members) > MAX_BYTES:
            raise ValueError("SDK archive exceeds bounded inventory")
        names = set()
        for member in members:
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or ".." in name.parts
                    or str(name) != member.name or member.name in names):
                raise ValueError("SDK archive contains unsafe or duplicate member")
            names.add(member.name)
        raw = bundle.extractfile("manifest.candidate.json").read()
        if hashlib.sha256(raw).hexdigest() != packet["manifest_sha256"]:
            raise ValueError("embedded SDK manifest digest differs from reviewed packet")
        manifest = json.loads(raw)
        if (manifest["schema"] != "tsotchke.eshkol.release.v1"
                or not VERSION.fullmatch(manifest["version"])
                or not OID.fullmatch(manifest["source_commit"])
                or manifest["source_commit"] != packet["source_commit"]
                or not HEX.fullmatch(manifest["build_id"])
                or manifest["build_id"] != packet["candidate_build_id"]):
            raise ValueError("SDK manifest source/version/build identity mismatch")
        rows = {}
        for name, row in manifest["artifacts"].items():
            path = row["path"]
            sha = row["sha256"]
            if PurePosixPath(path).is_absolute():
                if (name != "llvm_shared" or "LLVM" not in manifest.get("external_dependencies", {})
                        or not HEX.fullmatch(sha) or digest_file(path) != sha):
                    raise ValueError("SDK external LLVM identity mismatch")
                continue
            if not HEX.fullmatch(sha) or (path in rows and rows[path] != sha):
                raise ValueError("SDK manifest has conflicting artifact digests")
            rows[path] = sha
        if names != set(rows) | {"manifest.candidate.json", "LICENSE"}:
            raise ValueError("SDK archive and manifest inventories differ")
        for path, expected_sha in rows.items():
            if hashlib.file_digest(bundle.extractfile(path), "sha256").hexdigest() != expected_sha:
                raise ValueError(f"SDK artifact digest mismatch: {path}")
        compiler = bundle.extractfile(manifest["artifacts"]["compiler"]["path"]).read(8)
        if compiler != bytes.fromhex("cffaedfe0c000001"):
            raise ValueError("SDK compiler is not Mach-O arm64")
    return packet, manifest, archive


class GitHub:
    def json(self, endpoint):
        return json.loads(subprocess.check_output(["gh", "api", "--hostname", "github.com", endpoint], text=True))

    def upload(self, tag, archive):
        subprocess.run(["gh", "release", "upload", tag, str(archive), "--repo", REPOSITORY], check=True)

    def download(self, asset_id, destination):
        with destination.open("xb") as stream:
            subprocess.run(["gh", "api", f"repos/{REPOSITORY}/releases/assets/{asset_id}",
                            "--hostname", "github.com", "-H", "Accept: application/octet-stream"], stdout=stream, check=True)


def release_identity(api, tag, source):
    obj = api.json(f"repos/{REPOSITORY}/git/ref/tags/{tag}")["object"]
    for _ in range(8):
        if obj["type"] != "tag":
            break
        obj = api.json(f"repos/{REPOSITORY}/git/tags/{obj['sha']}")["object"]
    if obj["type"] != "commit" or obj["sha"] != source:
        raise ValueError("release tag does not resolve to reviewed SDK source")
    release = api.json(f"repos/{REPOSITORY}/releases/tags/{tag}")
    if release["draft"] or release["tag_name"] != tag:
        raise ValueError("SDK supplement requires an existing published release")
    return release


def prepare(packet_path, api):
    packet, manifest, archive = inspect_packet(packet_path)
    tag = "v" + manifest["version"]
    release = release_identity(api, tag, manifest["source_commit"])
    license_record = api.json(f"repos/{REPOSITORY}/contents/LICENSE?ref={tag}")
    if license_record.get("encoding") != "base64":
        raise ValueError("source license content unavailable")
    license_bytes = base64.b64decode(license_record["content"])
    with tarfile.open(archive, "r:gz") as bundle:
        if bundle.extractfile("LICENSE").read() != license_bytes:
            raise ValueError("SDK license differs from immutable source tag")
    sha = packet["archive"]["sha256"]
    name = f"eshkol-sdk-{tag}-darwin-arm64-{sha[:12]}.tar.gz"
    matching = [a for a in release["assets"] if a.get("digest") == "sha256:" + sha
                and a.get("size") == archive.stat().st_size]
    if len(matching) > 1:
        raise ValueError("multiple published SDK assets have the reviewed digest")
    if matching:
        name = matching[0]["name"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("unsafe SDK asset name")
    return {"schema": "eshkol.sdk.publisher-plan.v1", "repository": REPOSITORY,
            "tag": tag, "source_commit": manifest["source_commit"],
            "build_id": manifest["build_id"], "platform": packet["platform"],
            "archive_path": str(archive), "archive_sha256": sha,
            "archive_bytes": archive.stat().st_size, "manifest_sha256": packet["manifest_sha256"],
            "artifact_count": len(manifest["artifacts"]), "asset_name": name,
            "release_id": release["id"], "publisher_verified": False,
            "production_qualified": False}


def publish(plan, api, work_dir, expected_sha):
    if expected_sha != plan["archive_sha256"]:
        raise ValueError("publication requires the explicitly reviewed archive digest")
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=False)
    local = work / plan["asset_name"]
    shutil.copyfile(plan["archive_path"], local)
    if digest_file(local) != expected_sha:
        raise ValueError("staged upload digest changed")
    release = release_identity(api, plan["tag"], plan["source_commit"])
    if release["id"] != plan["release_id"]:
        raise ValueError("release identity changed since preparation")
    assets = [a for a in release["assets"] if a["name"] == plan["asset_name"]]
    if not assets:
        api.upload(plan["tag"], local)
        release = release_identity(api, plan["tag"], plan["source_commit"])
        assets = [a for a in release["assets"] if a["name"] == plan["asset_name"]]
    if len(assets) != 1:
        raise ValueError("SDK asset identity is missing or ambiguous")
    asset = assets[0]
    if (asset["state"] != "uploaded" or asset["size"] != plan["archive_bytes"]
            or asset.get("digest") != "sha256:" + expected_sha):
        raise ValueError("existing/uploaded SDK asset differs; replacement is forbidden")
    uploader = asset.get("uploader", {})
    if not uploader.get("login") or type(uploader.get("id")) is not int or uploader["id"] <= 0:
        raise ValueError("SDK asset has no authenticated publisher identity")
    downloaded = work / "authenticated-download.tar.gz"
    api.download(asset["id"], downloaded)
    if digest_file(downloaded) != expected_sha:
        raise ValueError("authenticated SDK download digest mismatch")
    receipt = {**plan, "schema": "eshkol.sdk.publisher-receipt.v1",
               "publisher_verified": True, "production_qualified": False,
               "asset_id": asset["id"], "asset_url": asset["browser_download_url"],
               "publisher": {"login": uploader["login"], "id": uploader["id"]},
               "retrieved_sha256": digest_file(downloaded)}
    (work / "publisher-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--work-dir")
    parser.add_argument("--expected-archive-sha256")
    args = parser.parse_args()
    if args.publish and (not args.work_dir or not args.expected_archive_sha256):
        parser.error("--publish requires --work-dir and --expected-archive-sha256")
    api = GitHub()
    plan = prepare(args.packet, api)
    result = publish(plan, api, args.work_dir, args.expected_archive_sha256) if args.publish else plan
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
