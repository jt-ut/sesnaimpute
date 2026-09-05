"""Private paged-ADQL streaming helper for the IRSA TAP catalogue pulls
under `sky/download`. Not the sibling `sky/download/_fetch.py` streamer,
which fetches one file by URL: an IRSA TAP pull is a query, so this helper
pages a `SELECT` by an ascending key column and streams each page's rows
to one CSV file, so a table with more rows than IRSA's per-query cap still
completes.
"""

import os
import urllib.parse
import urllib.request

SYNC_URL = "https://irsa.ipac.caltech.edu/TAP/sync"


def query_csv(table, columns, key_column, dest_path, page_size=200000, sync_url=SYNC_URL):
    """Streams `columns` from `table` to `dest_path` as CSV, keyset-paging
    ascending on `key_column` (must be `columns[0]`) in chunks of
    `page_size` rows. Returns `(n_rows, n_bytes)` written.
    """
    if columns[0] != key_column:
        raise ValueError("query_csv: key_column must be columns[0] to read the page cursor")
    col_list = ", ".join(columns)
    n_rows = 0
    last_key = 0
    header_written = False
    with open(dest_path, "w") as out:
        while True:
            adql = (
                f"SELECT TOP {page_size} {col_list} FROM {table} "
                f"WHERE {key_column} > {last_key} ORDER BY {key_column} ASC"
            )
            params = urllib.parse.urlencode(
                {"QUERY": adql, "FORMAT": "csv", "LANG": "ADQL"}
            ).encode("utf-8")
            with urllib.request.urlopen(sync_url, data=params, timeout=600) as resp:
                text = resp.read().decode("utf-8")
            lines = text.splitlines()
            if not lines:
                break
            header, body = lines[0], [ln for ln in lines[1:] if ln.strip()]
            if not body:
                break
            if not header_written:
                out.write(header + "\n")
                header_written = True
            out.write("\n".join(body) + "\n")
            n_rows += len(body)
            last_key = body[-1].split(",")[0]
            if len(body) < page_size:
                break
    n_bytes = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    return n_rows, n_bytes
