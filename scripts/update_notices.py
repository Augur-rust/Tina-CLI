"""Refresh notices from the locked Windows Cargo graph. Python 3.11+; no extra packages."""
import json
import pathlib
import shutil
import subprocess
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCOPE = ('Locked Windows x86_64 MSVC workspace dependencies with all features, including '
         'optional retrieval and build dependencies; a superset of the two shipped binaries. '
         'Model files, CUDA and Microsoft runtime installers are not bundled.')


def main():
    metadata = json.loads(subprocess.check_output([
        'cargo', 'metadata', '--locked', '--all-features', '--filter-platform',
        'x86_64-pc-windows-msvc', '--format-version', '1'
    ], cwd=ROOT, text=True, encoding='utf-8'))
    resolved = {node['id'] for node in metadata['resolve']['nodes']}
    checksums = {(p['name'], p['version']): p.get('checksum') for p in
                 tomllib.loads((ROOT / 'Cargo.lock').read_text('utf-8'))['package']}
    old = json.loads((ROOT / 'third-party/inventory.json').read_text('utf-8'))
    old_packages = {(p['name'], p['version']): p for p in old['packages']}
    packages = []
    for p in sorted(metadata['packages'], key=lambda p: (p['name'], p['version'])):
        if not p['source'] or p['id'] not in resolved:
            continue
        source = pathlib.Path(p['manifest_path']).parent
        notices = sorted(f for f in source.rglob('*') if f.is_file() and
                         f.name.upper().startswith(('LICENSE', 'LICENCE', 'COPYING', 'NOTICE', 'UNLICENSE', 'COPYRIGHT')))
        files = []
        for notice in notices:
            dest = ROOT / 'third-party/licenses' / (p['name'] + '-' + p['version']) / notice.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(notice, dest)
            files.append(dest.relative_to(ROOT).as_posix())
        if not files:
            # A matching upstream copy for usearch is already tracked with SOURCE.txt.
            previous = old_packages.get((p['name'], p['version']), {})
            files = previous.get('license_files', [])
            if not files or not all((ROOT / f).is_file() for f in files):
                raise RuntimeError('Missing upstream notices: ' + p['name'] + ' ' + p['version'])
        packages.append(dict(name=p['name'], version=p['version'], license=p['license'],
                             source=p['source'], registry_checksum_sha256=checksums[(p['name'], p['version'])],
                             license_files=files))
    inventory = dict(scope=SCOPE, source='cargo metadata --locked --all-features --filter-platform x86_64-pc-windows-msvc; Cargo.lock', packages=packages)
    (ROOT / 'third-party/inventory.json').write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8', newline='\n')
    lines = ['# Third-party dependency notices', '', SCOPE, '',
             'License and notice files come from the resolved Cargo registry packages. The USearch registry archive omits its license; its copy comes from the official matching version tag, recorded in SOURCE.txt. Text line endings may be normalized by Git. Machine-readable versions, registry checksums and source identifiers are in `third-party/inventory.json`.', '',
             'Regenerate with `python scripts/update_notices.py` (Python 3.11+).', '',
             '| Package | Version | Declared license | Notices |', '| --- | --- | --- | --- |']
    for p in packages:
        lines.append('| ' + ' | '.join([p['name'], p['version'], p['license'] or 'See notices', ', '.join('`' + f + '`' for f in p['license_files'])]) + ' |')
    (ROOT / 'THIRD_PARTY_NOTICES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print('Updated notices for', len(packages), 'resolved packages.')


if __name__ == '__main__':
    main()
