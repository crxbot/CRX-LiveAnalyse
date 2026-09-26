import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

FILENAME_RE = re.compile(r"^(?P<var_type>.+)_(?P<date>\d{8})_(?P<time>\d{4})\.webp$")

# ------------------------------
# Feste Geo-Referenzierung - aendert sich nie, daher hier hart hinterlegt
# ------------------------------
CRS = "EPSG:3857"
EXTENT_3857 = [
    160667.58918293554,
    5726764.0017245915,
    2087420.3952843477,
    7606221.471264458,
]

def scan_output_dir(output_dir: str):
    """Durchsucht output_dir (inkl. evtl. Unterordner je var_type) nach
    WEBPs und gruppiert die Timesteps pro var_type."""
    var_types: dict[str, set[str]] = {}

    for root, _dirs, files in os.walk(output_dir):
        for fname in files:
            m = FILENAME_RE.match(fname)
            if not m:
                continue
            var_type = m.group("var_type")
            ts = f"{m.group('date')}_{m.group('time')}"
            var_types.setdefault(var_type, set()).add(ts)

    return var_types


def load_existing_metadata(path: str | None) -> dict:
    """Liest eine vorher aus R2 heruntergeladene metadata.json ein.
    Existiert die Datei nicht (z.B. beim allerersten Lauf) oder ist sie
    kaputt, wird das wie 'kein bisheriger Stand' behandelt - es gibt dann
    einfach nichts zum Zusammenfuehren."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(
            f"Warnung: bestehende metadata.json ({path}) konnte nicht "
            f"gelesen werden, ignoriere sie: {e}",
            file=sys.stderr,
        )
        return {}


def merge_timesteps(existing_meta: dict, new_var_types_raw: dict, max_keep: int | None):
    """Fuehrt bestehende Timesteps (aus existing_meta['var_types']) und neu
    lokal gefundene (new_var_types_raw) pro var_type zusammen. Anschliessend
    wird - falls max_keep gesetzt ist - pro var_type auf die letzten
    max_keep Eintraege gekuerzt (aelteste zuerst raus).

    Gibt (var_types_out, removed_files) zurueck, wobei removed_files die
    Liste der dabei aussortierten WEBP-Dateinamen ist (fuers Loeschen in R2)."""
    existing_var_types = existing_meta.get("var_types", {})

    all_types = set(existing_var_types) | set(new_var_types_raw)

    var_types_out = {}
    removed_files = []

    for var_type in sorted(all_types):
        existing_ts = set(existing_var_types.get(var_type, {}).get("timesteps", []))
        new_ts = new_var_types_raw.get(var_type, set())
        combined = sorted(existing_ts | new_ts)

        if max_keep is not None and len(combined) > max_keep:
            dropped = combined[:-max_keep]
            kept = combined[-max_keep:]
        else:
            dropped = []
            kept = combined

        for ts in dropped:
            removed_files.append(f"{var_type}_{ts}.webp")

        var_types_out[var_type] = {
            "num_steps": len(kept),
            "timesteps": kept,
        }

    return var_types_out, removed_files


def build_meta(
    output_dir: str,
    run: str | None,
    date: str | None,
    existing_meta: dict,
    max_keep: int | None,
):
    var_types_raw = scan_output_dir(output_dir)

    if not var_types_raw:
        print(f"Keine passenden WEBPs in {output_dir} gefunden.", file=sys.stderr)

    var_types_out, removed_files = merge_timesteps(existing_meta, var_types_raw, max_keep)

    all_timesteps: list[str] = [
        ts for v in var_types_out.values() for ts in v["timesteps"]
    ]

    # date automatisch aus dem fruehesten Timestep ableiten, falls nicht
    # explizit uebergeben
    if not date:
        date = min(all_timesteps).split("_")[0] if all_timesteps else ""

    meta = {
        "run": run or "",
        "date": date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "crs": CRS,
        "extent_3857": EXTENT_3857,
        "var_types": var_types_out,
    }
    return meta, removed_files


def parse_args():
    parser = argparse.ArgumentParser(description="Baut metadata.json aus gerenderten WEBPs.")
    parser.add_argument("output_dir", help="Verzeichnis mit den gerenderten WEBPs")
    parser.add_argument("--run", default=None, help="Modelllauf, z.B. '18' (optional)")
    parser.add_argument("--date", default=None, help="Datum YYYYMMDD (optional, sonst automatisch)")
    parser.add_argument(
        "--existing",
        default=None,
        help="Pfad zu einer vorher aus R2 heruntergeladenen metadata.json (optional). "
        "Fehlt die Datei, wird ohne bisherigen Stand gestartet.",
    )
    parser.add_argument(
        "--max-keep",
        type=int,
        default=None,
        help="Maximal so viele Timesteps pro var_type behalten (aelteste zuerst raus). "
        "Ohne diese Option wird nicht gekuerzt.",
    )
    parser.add_argument(
        "--meta-out",
        default=None,
        help="Zielpfad fuer die metadata.json (Default: <parent von output_dir>/metadata.json)",
    )
    parser.add_argument(
        "--removed-out",
        default=None,
        help="Datei, in die entfernte Dateinamen (eine pro Zeile) geschrieben werden, "
        "damit der Workflow sie aus R2 loeschen kann.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    existing_meta = load_existing_metadata(args.existing)
    meta, removed_files = build_meta(
        args.output_dir, args.run, args.date, existing_meta, args.max_keep
    )

    meta_path = args.meta_out or os.path.join(
        os.path.dirname(args.output_dir.rstrip(os.sep)), "metadata.json"
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if args.removed_out:
        with open(args.removed_out, "w", encoding="utf-8") as f:
            if removed_files:
                f.write("\n".join(removed_files) + "\n")

    total_vars = len(meta["var_types"])
    total_files = sum(v["num_steps"] for v in meta["var_types"].values())
    print(f"metadata.json geschrieben: {meta_path} ({total_vars} var_types, {total_files} Dateien)")
    if removed_files:
        print(
            f"{len(removed_files)} Datei(en) sind aus dem Fenster gefallen "
            f"und sollten aus R2 geloescht werden: {', '.join(removed_files)}"
        )


if __name__ == "__main__":
    main()