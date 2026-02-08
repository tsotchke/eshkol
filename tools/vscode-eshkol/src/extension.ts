import * as path from 'path';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('eshkol');

    // Start LSP client if enabled
    if (config.get<boolean>('lsp.enabled', true)) {
        startLspClient(context, config);
    }

    // Register compile command
    context.subscriptions.push(
        vscode.commands.registerCommand('eshkol.compile', () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'eshkol') {
                vscode.window.showErrorMessage('No active Eshkol file');
                return;
            }
            const filePath = editor.document.fileName;
            const outputPath = filePath.replace(/\.esk$/, '');
            const compilerPath = getCompilerPath(config);

            const terminal = getOrCreateTerminal();
            terminal.show();
            terminal.sendText(`${compilerPath} "${filePath}" -o "${outputPath}"`);
        })
    );

    // Register compile-and-run command
    context.subscriptions.push(
        vscode.commands.registerCommand('eshkol.run', () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'eshkol') {
                vscode.window.showErrorMessage('No active Eshkol file');
                return;
            }
            const filePath = editor.document.fileName;
            const outputPath = filePath.replace(/\.esk$/, '');
            const compilerPath = getCompilerPath(config);

            const terminal = getOrCreateTerminal();
            terminal.show();
            terminal.sendText(`${compilerPath} "${filePath}" -o "${outputPath}" && "${outputPath}"`);
        })
    );

    // Register restart LSP command
    context.subscriptions.push(
        vscode.commands.registerCommand('eshkol.restartLsp', async () => {
            if (client) {
                await client.stop();
                client = undefined;
            }
            startLspClient(context, config);
            vscode.window.showInformationMessage('Eshkol Language Server restarted');
        })
    );
}

function startLspClient(
    context: vscode.ExtensionContext,
    config: vscode.WorkspaceConfiguration
) {
    const lspPath = getLspPath(config);

    const serverOptions: ServerOptions = {
        run: {
            command: lspPath,
            transport: TransportKind.stdio,
        },
        debug: {
            command: lspPath,
            transport: TransportKind.stdio,
        },
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'eshkol' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.esk'),
        },
    };

    client = new LanguageClient(
        'eshkol',
        'Eshkol Language Server',
        serverOptions,
        clientOptions
    );

    client.start();
    context.subscriptions.push({
        dispose: () => {
            if (client) {
                client.stop();
            }
        },
    });
}

function getLspPath(config: vscode.WorkspaceConfiguration): string {
    const configPath = config.get<string>('lsp.path', '');
    if (configPath) return configPath;
    return 'eshkol-lsp';
}

function getCompilerPath(config: vscode.WorkspaceConfiguration): string {
    const configPath = config.get<string>('compiler.path', '');
    if (configPath) return configPath;
    return 'eshkol-run';
}

let terminal: vscode.Terminal | undefined;

function getOrCreateTerminal(): vscode.Terminal {
    if (terminal && terminal.exitStatus === undefined) {
        return terminal;
    }
    terminal = vscode.window.createTerminal('Eshkol');
    return terminal;
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
