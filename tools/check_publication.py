#!/usr/bin/env python3
"""Check a research publication snapshot without printing sensitive values."""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import struct

IPV4 = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])')
HOME = re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+')
GPU = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
SECRET = re.compile(r'-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9_]{20,255}\b|\bgithub_pat_[A-Za-z0-9_]{20,255}\b|\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')
ASSIGNMENT = re.compile(r'''(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key|password|passwd)["']?\s*[:=]\s*["']([^"'\r\n]{10,})["']''')
EMAIL = re.compile(r'[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
SKIP_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache'}


def is_public_ip(value):
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return False


def png_chunks(data):
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Invalid PNG signature')
    offset = 8
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError('Truncated PNG chunk')
        size = struct.unpack('>I', data[offset:offset + 4])[0]
        end = offset + 12 + size
        if end > len(data):
            raise ValueError('Truncated PNG payload')
        kind = data[offset + 4:offset + 8]
        yield kind, data[offset:end]
        offset = end
    if offset != len(data):
        raise ValueError('Invalid PNG length')


def scan_text(text, private_terms=()):
    findings = []
    for label, pattern in [('credential', SECRET), ('credential_assignment', ASSIGNMENT),
                           ('personal_home_path', HOME), ('hardware_uuid', GPU)]:
        for match in pattern.finditer(text):
            findings.append({'rule': label, 'line': text.count('\n', 0, match.start()) + 1})
    for match in IPV4.finditer(text):
        if is_public_ip(match.group()):
            findings.append({'rule': 'public_ip', 'line': text.count('\n', 0, match.start()) + 1})
    for match in EMAIL.finditer(text):
        domain = match.group().split('@', 1)[1].lower()
        if domain not in {'users.noreply.github.com', 'example.com', 'example.org', 'example.net'}:
            findings.append({'rule': 'personal_email', 'line': text.count('\n', 0, match.start()) + 1})
    for term in private_terms:
        if len(term) < 3:
            continue
        for match in re.finditer(r'(?<![\w.-])' + re.escape(term) + r'(?![\w.-])', text, re.I):
            findings.append({'rule': 'private_identifier', 'line': text.count('\n', 0, match.start()) + 1})
    return findings


def safe_relative(value):
    path = Path(value)
    return bool(value) and not path.is_absolute() and '..' not in path.parts and '.git' not in path.parts


def check_tree(root, private_terms=()):
    root = Path(root).resolve()
    findings = []
    files = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts) or path.name == '.DS_Store':
            continue
        if path.is_symlink():
            findings.append({'path': str(relative), 'rule': 'symlink'})
            continue
        if not path.is_file():
            continue
        files.append(relative.as_posix())
        data = path.read_bytes()
        if path.suffix == '.png':
            for kind, _ in png_chunks(data):
                if kind in {b'tEXt', b'zTXt', b'iTXt', b'eXIf'}:
                    findings.append({'path': str(relative), 'rule': 'image_text_metadata'})
        else:
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError:
                findings.append({'path': str(relative), 'rule': 'unreviewed_binary'})
                continue
            findings.extend({'path': str(relative), **item} for item in scan_text(text, private_terms))
    manifest_path = root / 'PUBLICATION.json'
    if not manifest_path.is_file():
        findings.append({'path': 'PUBLICATION.json', 'rule': 'missing_manifest'})
        return findings
    manifest = json.loads(manifest_path.read_text())
    expected = {'PUBLICATION.json'}
    for item in manifest['files']:
        relative = item['path']
        if not safe_relative(relative):
            findings.append({'path': 'PUBLICATION.json', 'rule': 'unsafe_manifest_path'})
            continue
        expected.add(relative)
        path = root / relative
        if not path.is_file() or path.is_symlink():
            findings.append({'path': relative, 'rule': 'missing_file'})
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            findings.append({'path': relative, 'rule': 'checksum_mismatch'})
    for extra in set(files) - expected:
        findings.append({'path': extra, 'rule': 'unmanaged_file'})
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    findings = check_tree(args.root)
    print(json.dumps({'passed': not findings, 'findings': findings}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(findings))


if __name__ == '__main__':
    main()
