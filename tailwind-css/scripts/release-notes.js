import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as url from 'node:url';

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));

let version = process.argv[2] || process.env.npm_package_version;

if (!version) {
    let pkgPath = path.resolve(__dirname, '../packages/tailwindcss/package.json');
    let pkg = await fs.readFile(pkgPath, 'utf8').then(JSON.parse);

    version = pkg.version;
}

let changelog = await fs.readFile(path.resolve(__dirname, '..', '../CHANGELOG.md'), 'utf8');

let match = new RegExp(
    `## \\[${version}\\] - (.*)\\n\\n([\\s\\S]*?)\\n(?:(?:##\\s)|(?:\\[))`,
    'g'
).exec(changelog);

if (match) {
    let [, , notes] = match;

    console.log(notes.trim());
} else {
    console.log(`notas de versão do espaço reservado para a versão: v${version}`);
}