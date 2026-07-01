"""CLI do NVIDIA Startup AI Radar. Núcleo roda sem nenhuma chave de API."""
from __future__ import annotations

import argparse
import json
import sys

from .config import WEB_DIR, get_settings


def cmd_status(_):
    s = get_settings()
    print("NVIDIA Startup AI Radar — status")
    for k, v in s.status().items():
        print(f"  {k:16}: {v}")


def cmd_eval(_):
    from .eval.gold_eval import print_report
    print_report()


def cmd_calibrate(_):
    from .eval.calibrate import calibrate
    r = calibrate()
    print(f"Calibrado: native≥{r['native']} enabled≥{r['enabled']} "
          f"→ acurácia {r['tier_accuracy']*100:.0f}% (data/gold/calibration.json)")


def cmd_score(args):
    from .graph import run
    st = run(args.name)
    d = st.get("diagnosis")
    if not d:
        print("Erro:", "; ".join(st.get("errors", ["desconhecido"]))); return
    print(json.dumps(d.to_dict(), ensure_ascii=False, indent=2))


def cmd_brief(args):
    from .graph import run
    st = run(args.name)
    print(st.get("briefing") or "; ".join(st.get("errors", ["sem briefing"])))


def cmd_ask(args):
    from .rag.pipeline import ask
    for h in ask(args.query, top_k=args.k):
        print(f"[{h['score']}] {h['technology']}  <{h['source']}>")
        print(f"    {h['text'][:160]}")


def cmd_map(_):
    from .eval.gold_eval import map_data
    data = map_data()
    (WEB_DIR / "map_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (WEB_DIR / "map_data.js").write_text("window.__MAP_DATA = " + json.dumps(data, ensure_ascii=False) + ";", encoding="utf-8")
    print(f"Dado do Mapa de Alavancagem escrito em {WEB_DIR/'map_data.json'} (+ .js)")
    print(f"Abra {WEB_DIR/'index.html'} ou rode `radar serve`.")


def cmd_serve(args):
    from .api.app import serve
    serve(port=args.port)


def main(argv=None):
    p = argparse.ArgumentParser(prog="radar", description="NVIDIA Startup AI Radar")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="mostra camadas ativas").set_defaults(fn=cmd_status)
    sub.add_parser("eval", help="avalia o gold set (ranking/correlação)").set_defaults(fn=cmd_eval)
    sub.add_parser("calibrate", help="calibra limiares de tier pelo gold").set_defaults(fn=cmd_calibrate)
    sp = sub.add_parser("score", help="diagnóstico JSON de uma startup"); sp.add_argument("name"); sp.set_defaults(fn=cmd_score)
    sp = sub.add_parser("brief", help="briefing executivo de uma startup"); sp.add_argument("name"); sp.set_defaults(fn=cmd_brief)
    sp = sub.add_parser("ask", help="consulta o RAG da base NVIDIA"); sp.add_argument("query"); sp.add_argument("-k", type=int, default=4); sp.set_defaults(fn=cmd_ask)
    sub.add_parser("map", help="gera o dado do Mapa de Alavancagem").set_defaults(fn=cmd_map)
    sp = sub.add_parser("serve", help="sobe API + dashboard"); sp.add_argument("-p", "--port", type=int, default=8000); sp.set_defaults(fn=cmd_serve)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
