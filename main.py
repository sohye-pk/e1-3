import json
import re
import time

DATA_FILE = "data.json"
EPSILON = 1e-9
REPEAT_COUNT = 1000
SUPPORTED_SIZES = (3, 5, 13, 25)


def normalize_label(label):
    """외부 라벨을 Cross 또는 X로 정규화한다."""
    if not isinstance(label, str):
        return None

    label = label.strip().lower()

    if label in ("+", "cross"):
        return "Cross"

    if label == "x":
        return "X"

    return None


def validate_matrix(matrix, size):
    """N×N 숫자형 2차원 리스트인지 검증한다."""
    if not isinstance(matrix, list) or len(matrix) != size:
        return False

    for row in matrix:
        if not isinstance(row, list) or len(row) != size:
            return False

        for value in row:
            if isinstance(value, bool):
                return False

            if not isinstance(value, (int, float)):
                return False

    return True


def flatten_matrix(matrix):
    """2차원 배열을 1차원 배열로 변환한다."""
    result = []

    for row in matrix:
        for value in row:
            result.append(value)

    return result


# ==================================================
# MAC 연산
# ==================================================

def mac_2d(pattern, filter_matrix):
    """기존 2차원 배열 MAC 연산."""
    size = len(pattern)
    score = 0.0

    for row in range(size):
        for col in range(size):
            score += pattern[row][col] * filter_matrix[row][col]

    return score


def mac_1d(pattern, filter_matrix):
    """최적화된 1차원 배열 MAC 연산."""
    score = 0.0

    for index in range(len(pattern)):
        score += pattern[index] * filter_matrix[index]

    return score


# 최종 판정에는 최적화된 1D MAC 사용
def mac(pattern, filter_matrix):
    return mac_1d(pattern, filter_matrix)


# ==================================================
# 점수 비교
# ==================================================

def compare_scores(score_a, score_b, epsilon=EPSILON):
    """A, B 또는 UNDECIDED를 반환한다."""
    if abs(score_a - score_b) < epsilon:
        return "UNDECIDED"

    if score_a > score_b:
        return "A"

    return "B"


def compare_standard_scores(cross_score, x_score, epsilon=EPSILON):
    """Cross, X 또는 UNDECIDED를 반환한다."""
    if abs(cross_score - x_score) < epsilon:
        return "UNDECIDED"

    if cross_score > x_score:
        return "Cross"

    return "X"


# ==================================================
# 성능 측정
# ==================================================

def measure_mac(mac_function, pattern, filter_matrix,
                repeat=REPEAT_COUNT):
    """MAC 함수의 평균 실행 시간을 ms 단위로 측정한다."""

    start = time.perf_counter()

    for _ in range(repeat):
        mac_function(pattern, filter_matrix)

    elapsed = time.perf_counter() - start

    return elapsed / repeat * 1000


def measure_comparison(
    pattern_2d,
    filter_2d,
    pattern_1d,
    filter_1d
):
    """2D MAC과 1D MAC의 성능을 비교한다."""

    time_2d = measure_mac(
        mac_2d,
        pattern_2d,
        filter_2d
    )

    time_1d = measure_mac(
        mac_1d,
        pattern_1d,
        filter_1d
    )

    reduction = time_2d - time_1d

    if time_2d > 0:
        improvement = reduction / time_2d * 100
    else:
        improvement = 0.0

    return time_2d, time_1d, reduction, improvement


# ==================================================
# 사용자 입력
# ==================================================

def read_matrix_from_input(name, size):
    """N×N 행렬을 사용자로부터 입력받는다."""

    while True:
        print(f"{name} ({size}줄 입력, 공백 구분)")

        matrix = []
        valid = True

        for row_index in range(size):
            try:
                values = input(
                    f"{row_index + 1}행: "
                ).split()

                if len(values) != size:
                    print(
                        f"입력 형식 오류: 각 줄에 {size}개의 숫자를 "
                        "공백으로 구분해 입력하세요."
                    )
                    valid = False
                    break

                row = [float(value) for value in values]
                matrix.append(row)

            except ValueError:
                print(
                    f"입력 형식 오류: 각 줄에 {size}개의 숫자를 "
                    "공백으로 구분해 입력하세요."
                )
                valid = False
                break

        if valid:
            return matrix

        print("전체 행을 다시 입력하세요.\n")


# ==================================================
# 모드 1: 사용자 입력
# ==================================================

def run_user_mode():
    size = 3

    print("\n#----------------------------------------")
    print("# [1] 필터 입력")
    print("#----------------------------------------")

    # 2D 상태로 입력받음
    filter_a_2d = read_matrix_from_input(
        "필터 A",
        size
    )

    filter_b_2d = read_matrix_from_input(
        "필터 B",
        size
    )

    # 1D 배열로 한 번만 변환
    filter_a_1d = flatten_matrix(filter_a_2d)
    filter_b_1d = flatten_matrix(filter_b_2d)

    print("\n필터 A 저장 및 1차원 최적화 완료")
    print("필터 B 저장 및 1차원 최적화 완료")

    print("\n#----------------------------------------")
    print("# [2] 패턴 입력")
    print("#----------------------------------------")

    pattern_2d = read_matrix_from_input(
        "패턴",
        size
    )

    # 1D 배열로 한 번만 변환
    pattern_1d = flatten_matrix(pattern_2d)

    print("\n#----------------------------------------")
    print("# [3] MAC 결과")
    print("#----------------------------------------")

    # 실제 판정은 1D 최적화 MAC 사용
    score_a = mac_1d(
        pattern_1d,
        filter_a_1d
    )

    score_b = mac_1d(
        pattern_1d,
        filter_b_1d
    )

    result = compare_scores(
        score_a,
        score_b
    )

    print(f"A 점수: {score_a}")
    print(f"B 점수: {score_b}")

    if result == "UNDECIDED":
        print(
            f"판정: 판정 불가 "
            f"(|A-B| < {EPSILON})"
        )
    else:
        print(f"판정: {result}")

    print("\n#----------------------------------------")
    print("# [4] 성능 비교 (3×3)")
    print("#----------------------------------------")

    print("\n[A 필터 기준]")

    time_2d, time_1d, reduction, improvement = (
        measure_comparison(
            pattern_2d,
            filter_a_2d,
            pattern_1d,
            filter_a_1d
        )
    )

    print(f"기존 2D 방식: {time_2d:.6f} ms")
    print(f"최적화 1D 방식: {time_1d:.6f} ms")
    print(f"단축 시간: {reduction:.6f} ms")
    print(f"성능 변화: {improvement:.2f}%")


# ==================================================
# JSON 로드
# ==================================================

def load_json_data():
    """data.json을 읽는다."""

    try:
        with open(
            DATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

    except FileNotFoundError:
        print(
            f"오류: {DATA_FILE} 파일을 찾을 수 없습니다."
        )
        return None

    except json.JSONDecodeError as error:
        print(
            f"오류: JSON 형식이 잘못되었습니다. ({error})"
        )
        return None

    except OSError as error:
        print(
            f"오류: 파일을 읽을 수 없습니다. ({error})"
        )
        return None

    if not isinstance(data, dict):
        print(
            "오류: JSON 최상위 구조가 객체(dict)가 아닙니다."
        )
        return None

    return data


def extract_size_from_key(key):
    """size_13_1 형식의 키에서 13을 추출한다."""

    match = re.fullmatch(
        r"size_(\d+)_(\d+)",
        key
    )

    if not match:
        return None

    return int(match.group(1))


# ==================================================
# 필터 로드
# ==================================================

def load_filters(data):
    """
    JSON 필터를 검증한다.

    반환되는 필터는 2D, 1D 모두 보관한다.
    """

    filters = data.get("filters")

    if not isinstance(filters, dict):
        print(
            "오류: filters가 없거나 객체 형식이 아닙니다."
        )
        return {}

    normalized_filters = {}

    for size_key, filter_group in filters.items():

        match = re.fullmatch(
            r"size_(\d+)",
            size_key
        )

        if not match:
            print(
                f"[필터 무시] {size_key}: "
                "잘못된 필터 키 형식"
            )
            continue

        size = int(match.group(1))

        if size not in SUPPORTED_SIZES:
            print(
                f"[필터 무시] {size_key}: "
                "지원하지 않는 크기"
            )
            continue

        if not isinstance(filter_group, dict):
            print(
                f"[필터 오류] {size_key}: "
                "필터 그룹이 객체가 아닙니다."
            )
            continue

        normalized_group = {}
        group_valid = True

        for raw_label, matrix in filter_group.items():

            label = normalize_label(raw_label)

            if label is None:
                print(
                    f"[필터 오류] {size_key}: "
                    f"알 수 없는 라벨 '{raw_label}'"
                )
                group_valid = False
                continue

            # flatten 전에 2D 구조 검증
            if not validate_matrix(matrix, size):
                print(
                    f"[필터 오류] "
                    f"{size_key}/{raw_label}: "
                    f"{size}×{size} 배열이 아닙니다."
                )
                group_valid = False
                continue

            # 2D 원본과 1D 최적화 배열 모두 저장
            normalized_group[label] = {
                "2d": matrix,
                "1d": flatten_matrix(matrix)
            }

        if (
            "Cross" not in normalized_group
            or "X" not in normalized_group
        ):
            print(
                f"[필터 오류] {size_key}: "
                "Cross와 X 필터가 모두 필요합니다."
            )
            group_valid = False

        if group_valid:
            normalized_filters[size] = normalized_group

            print(
                f"✓ {size_key} 필터 로드 및 "
                "1차원 최적화 완료"
            )

        else:
            print(
                f"✗ {size_key} 필터 로드 실패"
            )

    return normalized_filters


# ==================================================
# 모드 2: JSON 패턴 분석
# ==================================================

def analyze_patterns(data, filters):
    """
    모든 패턴을 분석한다.

    2D 방식과 1D 방식의 판정 결과가 같은지도 확인한다.
    """

    patterns = data.get("patterns")

    if not isinstance(patterns, dict):
        print(
            "오류: patterns가 없거나 객체 형식이 아닙니다."
        )
        return [], 0, 0, {}

    results = []

    # 크기별 성능 측정용 데이터
    performance_cases = {}

    print("\n#----------------------------------------")
    print("# [2] 패턴 분석")
    print("#----------------------------------------")

    for case_id, case in patterns.items():

        print(f"\n--- {case_id} ---")

        try:
            size = extract_size_from_key(case_id)

            if size is None:
                raise ValueError(
                    "케이스 키가 "
                    "size_{N}_{idx} 형식이 아닙니다."
                )

            if size not in filters:
                raise ValueError(
                    f"size_{size} 필터를 사용할 수 없습니다."
                )

            if not isinstance(case, dict):
                raise ValueError(
                    "케이스 데이터가 객체 형식이 아닙니다."
                )

            pattern_2d = case.get("input")

            expected = normalize_label(
                case.get("expected")
            )

            if expected is None:
                raise ValueError(
                    "expected 값이 '+' 또는 'x' 형식의 "
                    "지원 라벨이 아닙니다."
                )

            # 2D 상태에서 먼저 검증
            if not validate_matrix(
                pattern_2d,
                size
            ):
                raise ValueError(
                    f"패턴 크기가 "
                    f"{size}×{size}와 일치하지 않습니다."
                )

            # 한 번만 1D 배열로 변환
            pattern_1d = flatten_matrix(
                pattern_2d
            )

            cross_filter_2d = (
                filters[size]["Cross"]["2d"]
            )

            x_filter_2d = (
                filters[size]["X"]["2d"]
            )

            cross_filter_1d = (
                filters[size]["Cross"]["1d"]
            )

            x_filter_1d = (
                filters[size]["X"]["1d"]
            )

            # 최적화된 1D MAC으로 실제 판정
            cross_score = mac_1d(
                pattern_1d,
                cross_filter_1d
            )

            x_score = mac_1d(
                pattern_1d,
                x_filter_1d
            )

            result = compare_standard_scores(
                cross_score,
                x_score
            )

            status = (
                "PASS"
                if result == expected
                else "FAIL"
            )

            print(
                f"Cross 점수: {cross_score}"
            )

            print(
                f"X 점수: {x_score}"
            )

            print(
                f"판정: {result} | "
                f"expected: {expected} | "
                f"{status}"
            )

            # 같은 크기의 첫 번째 패턴만 성능 측정용으로 저장
            if size not in performance_cases:
                performance_cases[size] = {
                    "pattern_2d": pattern_2d,
                    "pattern_1d": pattern_1d,
                    "filter_2d": cross_filter_2d,
                    "filter_1d": cross_filter_1d
                }

            if status == "FAIL":

                if result == "UNDECIDED":
                    reason = (
                        "동점(UNDECIDED) 처리 규칙"
                    )

                else:
                    reason = (
                        "판정 결과와 expected 불일치"
                    )

            else:
                reason = ""

        except (
            TypeError,
            ValueError,
            KeyError
        ) as error:

            status = "FAIL"
            reason = str(error)

            print(
                f"판정: FAIL ({reason})"
            )

        results.append(
            {
                "case_id": case_id,
                "status": status,
                "reason": reason
            }
        )

    total = len(results)

    passed = sum(
        1
        for result in results
        if result["status"] == "PASS"
    )

    return (
        results,
        total,
        passed,
        performance_cases
    )


# ==================================================
# 성능 비교 출력
# ==================================================

def print_performance_table(performance_cases):
    """2D 방식과 1D 방식의 성능을 비교 출력한다."""

    print(
        "크기       2D(ms)       1D(ms)       "
        "단축(ms)     변화율"
    )

    print(
        "------------------------------------------------------"
    )

    for size in sorted(performance_cases):

        case = performance_cases[size]

        time_2d, time_1d, reduction, improvement = (
            measure_comparison(
                case["pattern_2d"],
                case["filter_2d"],
                case["pattern_1d"],
                case["filter_1d"]
            )
        )

        print(
            f"{size}×{size:<6} "
            f"{time_2d:>10.6f}  "
            f"{time_1d:>10.6f}  "
            f"{reduction:>10.6f}  "
            f"{improvement:>7.2f}%"
        )


# ==================================================
# 모드 2 실행
# ==================================================

def run_json_mode():

    data = load_json_data()

    if data is None:
        return

    print("\n#----------------------------------------")
    print("# [1] 필터 로드")
    print("#----------------------------------------")

    filters = load_filters(data)

    (
        results,
        total,
        passed,
        performance_cases
    ) = analyze_patterns(
        data,
        filters
    )

    failed = total - passed

    print("\n#----------------------------------------")
    print("# [3] 성능 분석 (평균/1000회)")
    print("#----------------------------------------")

    print_performance_table(
        performance_cases
    )

    print("\n#----------------------------------------")
    print("# [4] 결과 요약")
    print("#----------------------------------------")

    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {failed}개")

    if failed:

        print("\n실패 케이스:")

        for result in results:

            if result["status"] == "FAIL":

                print(
                    f"- {result['case_id']}: "
                    f"{result['reason']}"
                )


# ==================================================
# 메인
# ==================================================

def main():

    print("=== Mini NPU Simulator ===")

    while True:

        print("\n[모드 선택]")
        print("1. 사용자 입력 (3×3)")
        print("2. data.json 분석")
        print("0. 종료")

        try:
            choice = input(
                "선택: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):
            print(
                "\n프로그램을 종료합니다."
            )
            return

        if choice == "1":
            run_user_mode()

        elif choice == "2":
            run_json_mode()

        elif choice == "0":
            print(
                "프로그램을 종료합니다."
            )
            return

        else:
            print(
                "입력 오류: "
                "0, 1, 2 중 하나를 선택하세요."
            )


if __name__ == "__main__":
    main()