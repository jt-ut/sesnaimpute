"""Replay the reviewed Musca inventory from retained original archive metadata.

This deliberately scopes the reviewed footprint selection to Musca. It does not
infer absence from missing downloads or from HPPSC source counts. The seven IDs
include the six intersecting observations plus the PCC550 co-map contributor.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import re

from . import maps

MUSCA_OBSIDS=('1342188099','1342188100','1342188101','1342216011',
              '1342216012','1342234098','1342234421')

@dataclass(frozen=True)
class InventoryProduct:
    logical_name: str
    obsids: tuple
    band: str
    mode: int
    level: str
    metadata: dict


def musca_inventory(config):
    """Return deduplicated selected products, requiring all reviewed raw inputs.

    JScanam L2.5 is preferred to its constituent L2 products. The standalone
    PCC550 observation has only L2. Scan/filter information comes from the
    retained observation-context FITS, never from the blue hardware label.
    """
    from astropy.io import fits
    from .download import _observation_xml
    root=maps.raw_root(config)
    with (root/'tap_pacs_musca_nearby_observations.csv').open(newline='') as stream:
        observations={r['observation_id']:r for r in csv.DictReader(stream)}
    products={}
    for obsid in MUSCA_OBSIDS:
        row=observations[obsid]
        xml=root/obsid/(obsid+'-herschel.ia.obs.ObservationContext-'+row['urn_version']+'.xml')
        envelope=_observation_xml(xml)
        contexts=list((root/obsid).glob('hpacs'+obsid+'obs*.fits.gz'))
        if len(contexts)!=1:
            raise ValueError('Require one retained observation context for '+obsid)
        header=fits.getheader(contexts[0])
        if str(header.get('OBS_ID',header.get('OBSID','')))!=obsid:
            raise ValueError('Observation context identity mismatch')
        band={'blue1':'P70','blue2':'P100'}.get(str(header.get('BLUEBAND','')).lower())
        if band is None:
            raise ValueError('Unverified observation filter for '+obsid)
        selected=[i for i in envelope.iter('item') if i.get('id')=='HPPJSMAPB']
        if not selected:
            selected=[i for i in envelope.iter('item') if i.get('id')=='HPPPMAPB']
        if not selected:
            raise ValueError('No reviewed map product for '+obsid)
        for item in selected:
            metadata={k:item.get(k,'') for k in ('id','urn','distribution_filename','distribution_path')}
            name=metadata['distribution_filename']
            logical=re.sub(r'_\d{13}\.fits\.gz$','',name)
            level='LEVEL2_5' if '/level2_5/' in metadata['distribution_path'].lower()+'/' else 'LEVEL2'
            previous=products.get(logical)
            members=(obsid,) if previous is None else tuple(sorted(set(previous.obsids+(obsid,))))
            if previous is not None and previous.band!=band:
                raise ValueError('Shared map has inconsistent filters')
            products[logical]=InventoryProduct(logical,members,band,int(row['observing_mode_oid']),
                level,metadata if previous is None else previous.metadata)
    return tuple(products[k] for k in sorted(products))


def rebuild_musca_index(config):
    """Index validated completed local products; return explicit pending entries.

    Partial files remain pending. This performs no network request and never
    promotes an incomplete inventory into a claim of complete regional coverage.
    """
    from .download import _validate_download,unpack_gzip
    expected=musca_inventory(config)
    root=maps.raw_root(config)
    complete={}
    unusable=[]
    # Include the retained IRSA equivalent, whose delivered filename differs.
    for obsid in MUSCA_OBSIDS:
        for path in sorted((root/obsid).rglob('*.fits')):
            if 'level2' not in str(path.parent).lower():
                continue
            _validate_download(path,path)
            for product in expected:
                if obsid not in product.obsids:
                    continue
                try:
                    record=maps.inspect_map(config,path,obsids=product.obsids,
                        mode=product.mode,level=product.level)
                except ValueError as exc:
                    unusable.append(dict(path=str(path),logical_name=product.logical_name,
                                         reason=str(exc)))
                    continue
                if record.map_id==product.logical_name:
                    complete[record.map_id]=record
                    break
    for product in expected:
        if product.logical_name in complete:
            continue
        info=product.metadata
        path=maps.checked_path(root,str(Path(info['distribution_path'].lstrip('/'))/info['distribution_filename']))
        if path.exists():
            _validate_download(path,path)
            record=maps.inspect_map(config,unpack_gzip(path,max_bytes=1024**3),
                obsids=product.obsids,mode=product.mode,level=product.level)
            if record.map_id!=product.logical_name:
                raise ValueError('Downloaded map identity disagrees with retained inventory')
            complete[record.map_id]=record
    pending=tuple(p for p in expected if p.logical_name not in complete)
    if not complete:
        raise RuntimeError('No complete validated Musca maps; existing index preserved')
    maps.write_index(config,'Musca',list(complete.values()))
    return dict(region='Musca',records=tuple(complete.values()),pending=pending,
                complete=not pending,unusable=tuple(unusable))
