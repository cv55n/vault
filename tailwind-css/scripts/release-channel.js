// 1.2.3                 -> latest (padrão)
// 0.0.0-insiders.ffaa88 -> insiders
// 4.1.0-alpha.4         -> alpha

let version =
    process.argv[2] ||
    process.env.npm_package_version ||
    require('./packages/tailwindcss/package.json').version;

let match = /\d+\.\d+\.\d+-(.*)\.\d+/g.exec(version);

if (match) {
    console.log(match[1]);
} else {
    console.log('latest');
}