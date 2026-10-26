"""PACS archive transport and local regional-index assembly.

Official endpoints: astroquery.esa.hsa's documented HSA transport, and
https://archives.esac.esa.int/hsa/aio/doc/datarequests.html .
Completed files are authoritative. Partial transfers have a .part suffix.
"""
from pathlib import Path
from contextlib import contextmanager
import fcntl
import gzip
import os
import re
import shutil
import tarfile
import time

import urllib.error
import urllib.parse
import urllib.request
from sesnaimpute import config as config_module, regions as regions_module, progress
from . import maps

TAP_URL = 'https://archives.esac.esa.int/hsa/whsa-tap-server/tap/sync'
DATA_URL = 'https://archives.esac.esa.int/hsa/whsa-tap-server/data'
MAX_REGION_BYTES = 10 * 1024**3
CHUNK_BYTES = 1024**2


@contextmanager
def _partial_writer(path):
    """Exclusive ownership without a persistent lock/provenance sidecar."""
    path = Path(path)
    maps.checked_path(path.parent, path.name)
    if path.exists():
        yield None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = maps.checked_path(path.parent, path.name + '.part')
    # a+b opens without truncating before the lock has been acquired.
    with partial.open('a+b') as out:
        try:
            fcntl.flock(out.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another writer owns partial product: {partial}')
        try:
            if path.exists():
                yield None
                return
            if os.fstat(out.fileno()).st_ino != partial.stat().st_ino:
                raise RuntimeError('Partial product changed while acquiring ownership')
            yield out
        finally:
            fcntl.flock(out.fileno(), fcntl.LOCK_UN)


def _validate_download(partial, target):
    """Reject service error bodies; check complete compressed streams before publish."""
    name = str(target).lower()
    if name.endswith('.fits.gz'):
        with gzip.open(partial, 'rb') as stream:
            if not stream.read(80).startswith(b'SIMPLE  ='):
                raise ValueError('Downloaded gzip is not FITS')
            size = 80
            while True:
                block = stream.read(CHUNK_BYTES)
                if not block:
                    break
                size += len(block)
                if size > MAX_REGION_BYTES:
                    raise ValueError('Uncompressed FITS exceeds validation budget')
        if size % 2880:
            raise ValueError('FITS length is not a multiple of 2880')
    elif name.endswith('.fits'):
        from astropy.io import fits
        import warnings
        size = Path(partial).stat().st_size
        if not size or size % 2880:
            raise ValueError('Invalid FITS file length')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with fits.open(partial, mode='readonly', memmap=True,
                           lazy_load_hdus=False) as hdus:
                hdus.verify('exception')
                end = 0
                for hdu in hdus:
                    info = hdu.fileinfo()
                    end = max(end, info['datLoc'] + info['datSpan'])
                if end != size:
                    raise ValueError('FITS HDU extents do not match file length')
    elif name.endswith('.tar'):
        with tarfile.open(partial) as archive:
            if not archive.getmembers():
                raise ValueError('Empty archive response')
    elif name.endswith('.csv'):
        import csv
        with open(partial, newline='') as stream:
            first = stream.readline()
            if not first.strip() or first.lstrip().startswith('<'):
                raise ValueError('Expected CSV response, received service error/invalid header')
            width = len(next(csv.reader([first])))
            if any(len(row) != width for row in csv.reader(stream)):
                raise ValueError('Malformed CSV response')
    elif name.endswith('.xml'):
        _observation_xml(partial)
    else:
        raise ValueError(f'No integrity validator for archive response: {target}')


def _promote(path, out):
    """Called under partial-file lock; completed files never share a mutable writer."""
    out.flush()
    partial = Path(str(path) + '.part')
    os.link(partial, path)
    partial.unlink()


def fetch(url, path, *, params=None, max_bytes=MAX_REGION_BYTES,
          max_seconds=900):
    """Stream under exclusive partial ownership, checking transport and file integrity.

    A valid 416 response can confirm a fully transferred partial. An ignored
    Range restarts only the exclusively held partial file. Uses the archived
    HGBS urllib transport with 1 MiB progress reads and a 600-second socket timeout.
    The overall time budget is checked between blocking reads.
    """
    path = Path(path)
    with _partial_writer(path) as out:
        if out is None:
            return path
        out.seek(0, os.SEEK_END)
        offset = out.tell()
        if offset > max_bytes:
            raise RuntimeError('Partial transfer exceeds byte limit')
        headers = {'Accept-Encoding': 'identity',
                   'User-Agent': 'sesna-complete/census fetch_hgbs (research; anonymous)'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        started = time.monotonic()
        if params:
            parsed = urllib.parse.urlsplit(url)
            query = urllib.parse.urlencode(params, doseq=True)
            url = urllib.parse.urlunsplit(parsed._replace(
                query='&'.join(part for part in (parsed.query, query) if part)))
        request = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=600)
        except urllib.error.HTTPError as exc:
            if exc.code != 416:
                exc.close()
                raise
            response = exc
        with response:
            status = response.status
            print('PACS transfer response:', status,
                  {k: response.headers.get(k) for k in
                   ('Content-Length', 'Content-Range', 'Accept-Ranges',
                    'Content-Encoding', 'Transfer-Encoding')}, flush=True)
            if status == 416:
                match = re.fullmatch(r'bytes \*/(\d+)',
                                     response.headers.get('Content-Range', ''))
                if match and offset > 0 and int(match[1]) == offset:
                    out.flush()
                    _validate_download(Path(str(path) + '.part'), path)
                    _promote(path, out)
                    return path
                raise RuntimeError('Range unsatisfied and partial completeness unproven')
            if response.headers.get('Content-Encoding', 'identity') != 'identity':
                raise RuntimeError('Encoded HTTP body cannot be resumed safely')
            if status == 206:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                     response.headers.get('Content-Range', ''))
                if not match or int(match[1]) != offset:
                    raise RuntimeError('Server returned inconsistent Content-Range')
                total = int(match[3])
                if int(match[2]) != total - 1 or total <= offset:
                    raise RuntimeError('Server did not return requested remaining range')
            elif status == 200:
                if offset:
                    out.seek(0)
                    out.truncate(0)
                    offset = 0
                total = int(response.headers['Content-Length']) if 'Content-Length' in response.headers else None
            else:
                raise RuntimeError(f'Unexpected download status {status}')
            if total is not None and total > max_bytes:
                raise RuntimeError('Archive exceeds byte limit')
            remaining = shutil.disk_usage(path.parent).free
            if remaining < (total - offset if total is not None else CHUNK_BYTES):
                raise RuntimeError('Insufficient disk space for archive transfer')
            count = offset
            while True:
                if time.monotonic() - started > max_seconds:
                    raise RuntimeError('Transfer time budget exceeded; partial retained')
                block = response.read(1024**2)
                if time.monotonic() - started > max_seconds:
                    raise RuntimeError('Transfer time budget exceeded; partial retained')
                if not block:
                    break
                if count + len(block) > max_bytes:
                    raise RuntimeError('Transfer byte budget exceeded; partial retained')
                if shutil.disk_usage(path.parent).free < len(block):
                    raise RuntimeError('Disk full; partial retained')
                out.write(block)
                count += len(block)
                print(f'PACS transfer: {count:,}'
                      + (f'/{total:,}' if total is not None else '')
                      + f' bytes, {time.monotonic() - started:.1f}s', flush=True)
            if total is not None and count != total:
                raise RuntimeError('Incomplete transfer; partial retained')
        out.flush()
        _validate_download(Path(str(path) + '.part'), path)
        _promote(path, out)
    return path


def query_csv(config, filename, query):
    """Cache the archive's verbatim response; never re-query a present file."""
    if Path(filename).name != filename or not filename.endswith('.csv'):
        raise ValueError('Raw TAP cache must be a simple .csv filename')
    return fetch(TAP_URL, maps.checked_path(maps.raw_root(config), filename),
                 params={'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv',
                         'QUERY': query}, max_bytes=256 * 1024**2)


def fetch_observation(config, obsid, level='LEVEL2_5', *, max_bytes=MAX_REGION_BYTES):
    obsid = str(obsid)
    if not obsid.isdigit():
        raise ValueError('Observation identifier must be numeric')
    if level not in ('LEVEL2', 'LEVEL2_5', 'LEVEL3'):
        raise ValueError('Only calibrated map levels may be fetched')
    path = maps.checked_path(maps.raw_root(config), f'{obsid}/{obsid}_{level}.tar')
    return fetch(DATA_URL, path, params={'retrieval_type': 'OBSERVATION',
                 'observation_id': obsid, 'instrument_name': 'PACS',
                 'product_level': level}, max_bytes=max_bytes)


def extract_selected(archive, destination, keep, *, max_bytes=MAX_REGION_BYTES):
    """Extract selected regular members safely; never overwrite existing bytes.

    `keep(member_name)` must select inspected blue-channel maps and required
    metadata. Links, devices, traversal, and absolute paths are rejected.
    No preexisting archive or product is deleted. Gzip conversion is separate.
    """
    destination = Path(destination)
    written = []
    total = 0
    with tarfile.open(archive, 'r:*') as tar:
        for member in tar:
            parts = Path(member.name).parts
            if Path(member.name).is_absolute() or '..' in parts:
                raise ValueError(f'Unsafe archive member: {member.name}')
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f'Non-regular archive member: {member.name}')
            if not keep(member.name):
                continue
            path = maps.checked_path(destination, member.name)
            if path.exists():
                continue
            total += member.size
            if total > max_bytes:
                raise RuntimeError('Selected archive members exceed extraction budget')
            path.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(path.parent).free < member.size:
                raise RuntimeError('Insufficient disk space to unpack archive')
            with _partial_writer(path) as output:
                if output is None:
                    continue
                output.seek(0)
                output.truncate(0)
                with tar.extractfile(member) as source:
                    shutil.copyfileobj(source, output, CHUNK_BYTES)
                output.flush()
                if output.tell() != member.size:
                    raise RuntimeError('Incomplete archive member; partial retained')
                _promote(path, output)
            written.append(path)
    return written


def unpack_gzip(path, *, max_bytes=MAX_REGION_BYTES):
    """Retain original bytes and unpack one map for genuine FITS memory mapping."""
    path = Path(path)
    if path.suffix != '.gz':
        return path
    output = maps.checked_path(path.parent, path.stem)
    if output.exists():
        return output
    with _partial_writer(output) as target:
        if target is None:
            return output
        target.seek(0)
        target.truncate(0)
        count = 0
        with gzip.open(path, 'rb') as source:
            while True:
                block = source.read(CHUNK_BYTES)
                if not block:
                    break
                count += len(block)
                if count > max_bytes or shutil.disk_usage(path.parent).free < len(block):
                    raise RuntimeError('Uncompressed map exceeds available space/budget')
                target.write(block)
        _promote(output, target)
    return output


def _observation_xml(path):
    """Validate the actual HSA observation request envelope, not just XML syntax."""
    from xml.etree import ElementTree
    root = ElementTree.parse(path).getroot()
    if (root.tag != 'request' or root.get('instrument') != 'PACS'
            or not root.get('id', '').isdigit()
            or not root.get('urn', '').startswith('urn:hsa:herschel.ia.obs.ObservationContext:')):
        raise ValueError('Expected an HSA PACS observation request XML, not a service error')
    return root


def observation_products(xml_path):
    """Read map product identifiers from an original archive request XML.

    Select only blue JScanam/Unimap products. Product wavelengths are confirmed
    from FITS metadata after download; the blue camera alone does not mean P70.
    """
    products = []
    for item in _observation_xml(xml_path).iter('item'):
        if item.get('id') not in ('HPPUNIMAPB', 'HPPJSMAPB'):
            continue
        products.append({key: item.get(key, '') for key in
                         ('id', 'urn', 'distribution_filename', 'distribution_path')})
    return products


def fetch_product(config, product, *, max_bytes=MAX_REGION_BYTES, max_seconds=900):
    """Fetch one XML-identified map, preserving its archive delivery path/name."""
    name = product['distribution_filename']
    if Path(name).name != name:
        raise ValueError('Unsafe archive distribution filename')
    relative = str(Path(product['distribution_path'].lstrip('/')) / name)
    path = maps.checked_path(maps.raw_root(config), relative)
    return fetch(DATA_URL, path,
                 params={'retrieval_type': 'PRODUCT', 'hcss_urn': product['urn']},
                 max_bytes=max_bytes, max_seconds=max_seconds)


def build(config, regions=None):
    """Rebuild reviewed Musca discovery/index from retained archive metadata.

    Missing products are reported explicitly; this stage does not silently
    reinterpret transport failures as absent coverage or launch bulk transfers.
    Other regions require their own verified footprint inventory first.
    """
    from .inventory import rebuild_musca_index
    names=['Musca'] if regions is None else ([regions] if isinstance(regions,str) else list(regions))
    names=[getattr(r,'name',r) for r in names]
    if any(name!='Musca' for name in names):
        raise NotImplementedError('Only the retained Musca inventory is currently reviewed')
    results=[]
    for name in names:
        result=rebuild_musca_index(config)
        print('PACS inventory '+name+': '+str(len(result['records']))+' complete maps; '+
              str(len(result['pending']))+' pending products',flush=True)
        for item in result['unusable']:
            print('PACS unusable local map '+item['path']+': '+item['reason'],flush=True)
        for product in result['pending']:
            print('PACS pending '+product.band+' '+product.logical_name+' '+product.metadata['urn'],flush=True)
        results.append(result)
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    build(config_module.load(args.config))
