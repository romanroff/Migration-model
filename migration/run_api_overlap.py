import argparse

from api_overlap import run_api_overlap_pipeline
from config import API_OVERLAP_MODEL_PATH, LOOP_REQUEST_SLEEP_SECONDS, TOP_PERCENT


def parse_args():
    parser = argparse.ArgumentParser(description="Directed OD-прогноз по API main_info")
    parser.add_argument("territory_id", type=int, help="ID верхнеуровневой территории")
    parser.add_argument("--down-by", type=int, default=1, help="Уровень детализации дочерних территорий")
    parser.add_argument("--loop", action="store_true", help="Собрать второй уровень по всем territory_id первого уровня")
    parser.add_argument(
        "--loop-request-sleep-seconds",
        type=float,
        default=LOOP_REQUEST_SLEEP_SECONDS,
        help="Пауза между запросами второго уровня в секундах",
    )
    parser.add_argument("--top-percent", type=int, default=TOP_PERCENT, help="Доля рёбер для карты в процентах")
    parser.add_argument("--top-n", type=int, default=None, help="Абсолютное количество рёбер для карты")
    parser.add_argument(
        "--model-path",
        type=str,
        default=API_OVERLAP_MODEL_PATH,
        help="Путь до overlap-модели",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Каталог для CSV и HTML-карты",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_api_overlap_pipeline(
        territory_id=args.territory_id,
        down_by=args.down_by,
        loop=args.loop,
        loop_request_sleep_seconds=args.loop_request_sleep_seconds,
        top_percent=args.top_percent,
        top_n=args.top_n,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )

    print(f"LOOP: {result['loop']}")
    if result["loop"]:
        print(f"Территорий первого уровня: {len(result['first_level_ids'])}")
    print(f"Всего API-запросов: {result['request_count']}")
    print(f"Территорий: {len(result['territories'])}")
    print(f"OD-пар: {len(result['predictions'])}")
    print(f"CSV сохранён: {result['predictions_path']}")
    print(f"Карта сохранена: {result['map_path']}")


if __name__ == "__main__":
    main()
