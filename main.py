import json
import re
import time

DATA_FILE = "data.json"
EPSILON = 1e-9
REPEAT_COUNT = 10
SUPPORTED_SIZES = (3, 5, 13, 25)


def normalize_label(label):
    """Convert external labels to the two internal standard labels."""
    if not isinstance(label, str):
        return None

    label = label.strip().lower()

    if label in ("+", "cross"):
        return "Cross"
    if label == "x":
        return "X"

    return None


def validate_matrix(matrix, size):
    """Validate that matrix is an N x N numeric 2D list."""
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


def mac(pattern, filter_matrix):
    """Perform MAC using explicit nested loops. No external libraries."""
    size = len(pattern)
    score = 0.0

    for row in range(size):
        for col in range(size):
            score += pattern[row][col] * filter_matrix[row][col]

    return score


def compare_scores(score_a, score_b, epsilon=EPSILON):
    """Return A, B, or UNDECIDED according to epsilon policy."""
    difference = abs(score_a - score_b)

    if difference < epsilon:
        return "UNDECIDED"
    if score_a > score_b:
        return "A"
    return "B"


def compare_standard_scores(cross_score, x_score, epsilon=EPSILON):
    """Return Cross, X, or UNDECIDED according to epsilon policy."""
    difference = abs(cross_score - x_score)

    if difference < epsilon:
        return "UNDECIDED"
    if cross_score > x_score:
        return "Cross"
    return "X"


def measure_mac(pattern, filter_matrix, repeat=REPEAT_COUNT):
    """Measure only the MAC function call, excluding I/O."""
    start = time.perf_counter()

    for _ in range(repeat):
        mac(pattern, filter_matrix)

    elapsed = time.perf_counter() - start
    average_ms = elapsed / repeat * 1000

    return average_ms


def read_matrix_from_input(name, size):
    """Read an N x N matrix one row at a time with validation."""
    while True:
        print(f"{name} ({size}줄 입력, 공백 구분)")

        matrix = []
        valid = True

        for row_index in range(size):
            try:
                values = input(f"{row_index + 1}행: ").split()

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

        # 이미 일부 행을 입력한 상태이므로 전체 행을 다시 입력받는다.
        print("전체 행을 다시 입력하세요.\n")


def run_user_mode():
    size = 3

    print("\n#----------------------------------------")
    print("# [1] 필터 입력")
    print("#----------------------------------------")

    filter_a = read_matrix_from_input("필터 A", size)
    filter_b = read_matrix_from_input("필터 B", size)

    print("\n필터 A 저장 완료")
    print("필터 B 저장 완료")

    print("\n#----------------------------------------")
    print("# [2] 패턴 입력")
    print("#----------------------------------------")

    pattern = read_matrix_from_input("패턴", size)

    print("\n#----------------------------------------")
    print("# [3] MAC 결과")
    print("#----------------------------------------")

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    average_ms = measure_mac(pattern, filter_a)

    result = compare_scores(score_a, score_b)

    print(f"A 점수: {score_a}")
    print(f"B 점수: {score_b}")
    print(f"연산 시간(평균/{REPEAT_COUNT}회): {average_ms:.6f} ms")

    if result == "UNDECIDED":
        print(f"판정: 판정 불가 (|A-B| < {EPSILON})")
    else:
        print(f"판정: {result}")

    print("\n#----------------------------------------")
    print("# [4] 성능 분석 (3×3)")
    print("#----------------------------------------")
    print_performance_table({size: (pattern, filter_a)})


def load_json_data():
    """Load data.json. Return parsed data or None on file-level failure."""
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

    except FileNotFoundError:
        print(f"오류: {DATA_FILE} 파일을 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError as error:
        print(f"오류: JSON 형식이 잘못되었습니다. ({error})")
        return None
    except OSError as error:
        print(f"오류: 파일을 읽을 수 없습니다. ({error})")
        return None

    if not isinstance(data, dict):
        print("오류: JSON 최상위 구조가 객체(dict)가 아닙니다.")
        return None

    return data


def extract_size_from_key(key):
    """Extract N from keys such as size_13_1."""
    match = re.fullmatch(r"size_(\d+)_(\d+)", key)

    if not match:
        return None

    return int(match.group(1))


def load_filters(data):
    """Load and validate filters while normalizing cross/x labels."""
    filters = data.get("filters")

    if not isinstance(filters, dict):
        print("오류: filters가 없거나 객체 형식이 아닙니다.")
        return {}

    normalized_filters = {}

    for size_key, filter_group in filters.items():
        match = re.fullmatch(r"size_(\d+)", size_key)

        if not match:
            print(f"[필터 무시] {size_key}: 잘못된 필터 키 형식")
            continue

        size = int(match.group(1))

        if size not in SUPPORTED_SIZES:
            print(f"[필터 무시] {size_key}: 지원하지 않는 크기")
            continue

        if not isinstance(filter_group, dict):
            print(f"[필터 오류] {size_key}: 필터 그룹이 객체가 아닙니다.")
            continue

        normalized_group = {}
        group_valid = True

        for raw_label, matrix in filter_group.items():
            label = normalize_label(raw_label)

            if label is None:
                print(
                    f"[필터 오류] {size_key}: 알 수 없는 라벨 '{raw_label}'"
                )
                group_valid = False
                continue

            if not validate_matrix(matrix, size):
                print(
                    f"[필터 오류] {size_key}/{raw_label}: "
                    f"{size}×{size} 배열이 아닙니다."
                )
                group_valid = False
                continue

            normalized_group[label] = matrix

        if "Cross" not in normalized_group or "X" not in normalized_group:
            print(
                f"[필터 오류] {size_key}: Cross와 X 필터가 모두 필요합니다."
            )
            group_valid = False

        if group_valid:
            normalized_filters[size] = normalized_group
            print(f"✓ {size_key} 필터 로드 완료 (Cross, X)")
        else:
            print(f"✗ {size_key} 필터 로드 실패")

    return normalized_filters


def analyze_patterns(data, filters):
    """Analyze every pattern independently so one bad case cannot stop others."""
    patterns = data.get("patterns")

    if not isinstance(patterns, dict):
        print("오류: patterns가 없거나 객체 형식이 아닙니다.")
        return [], 0, 0

    results = []

    print("\n#----------------------------------------")
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#----------------------------------------")

    for case_id, case in patterns.items():
        print(f"\n--- {case_id} ---")

        total_tests = 1
        passed = 0

        try:
            size = extract_size_from_key(case_id)

            if size is None:
                raise ValueError("케이스 키가 size_{N}_{idx} 형식이 아닙니다.")

            if size not in filters:
                raise ValueError(f"size_{size} 필터를 사용할 수 없습니다.")

            if not isinstance(case, dict):
                raise ValueError("케이스 데이터가 객체 형식이 아닙니다.")

            pattern = case.get("input")
            expected = normalize_label(case.get("expected"))

            if expected is None:
                raise ValueError(
                    "expected 값이 '+' 또는 'x' 형식의 지원 라벨이 아닙니다."
                )

            if not validate_matrix(pattern, size):
                raise ValueError(
                    f"패턴 크기가 {size}×{size}와 일치하지 않습니다."
                )

            cross_filter = filters[size]["Cross"]
            x_filter = filters[size]["X"]

            if not validate_matrix(cross_filter, size):
                raise ValueError("Cross 필터 크기가 패턴과 일치하지 않습니다.")

            if not validate_matrix(x_filter, size):
                raise ValueError("X 필터 크기가 패턴과 일치하지 않습니다.")

            cross_score = mac(pattern, cross_filter)
            x_score = mac(pattern, x_filter)
            result = compare_standard_scores(cross_score, x_score)

            status = "PASS" if result == expected else "FAIL"

            if status == "PASS":
                passed = 1

            print(f"Cross 점수: {cross_score}")
            print(f"X 점수: {x_score}")
            print(
                f"판정: {result} | expected: {expected} | {status}"
            )

            if status == "FAIL":
                if result == "UNDECIDED":
                    reason = "동점(UNDECIDED) 처리 규칙"
                else:
                    reason = "판정 결과와 expected 불일치"

            else:
                reason = ""

        except (TypeError, ValueError, KeyError) as error:
            status = "FAIL"
            reason = str(error)
            print(f"판정: FAIL ({reason})")

        results.append(
            {
                "case_id": case_id,
                "status": status,
                "reason": reason,
            }
        )

    return results, len(results), sum(
        1 for result in results if result["status"] == "PASS"
    )


def print_performance_table(cases):
    """Print average MAC time for each matrix size."""
    print("크기       평균 시간(ms)    연산 횟수")
    print("-------------------------------------")

    for size in sorted(cases):
        pattern, filter_matrix = cases[size]
        average_ms = measure_mac(pattern, filter_matrix)

        print(
            f"{size}×{size:<6} "
            f"{average_ms:>12.6f}    {size * size}"
        )


def run_json_mode():
    data = load_json_data()

    if data is None:
        return

    print("\n#----------------------------------------")
    print("# [1] 필터 로드")
    print("#----------------------------------------")

    filters = load_filters(data)

    results, total, passed = analyze_patterns(data, filters)
    failed = total - passed

    print("\n#----------------------------------------")
    print("# [3] 성능 분석 (평균/10회)")
    print("#----------------------------------------")

    performance_cases = {}

    # data.json의 실제 패턴을 사용할 수 있는 크기는 그대로 사용한다.
    patterns = data.get("patterns", {})

    for size in SUPPORTED_SIZES:
        filter_group = filters.get(size)

        if filter_group is None:
            continue

        pattern = None

        for case_id, case in patterns.items():
            if extract_size_from_key(case_id) == size:
                if isinstance(case, dict) and validate_matrix(case.get("input"), size):
                    pattern = case["input"]
                    break

        if pattern is None:
            # 3x3은 JSON에 없어도 성능 분석 요구사항상 측정할 수 있도록
            # 0으로 채운 기본 패턴을 사용한다.
            if size == 3:
                pattern = [[0.0] * size for _ in range(size)]
            else:
                continue

        performance_cases[size] = (pattern, filter_group["Cross"])

    print_performance_table(performance_cases)

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
                print(f"- {result['case_id']}: {result['reason']}")


def main():
    print("=== Mini NPU Simulator ===")

    while True:
        print("\n[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        print("0. 종료")

        try:
            choice = input("선택: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n프로그램을 종료합니다.")
            return

        if choice == "1":
            run_user_mode()

        elif choice == "2":
            run_json_mode()

        elif choice == "0":
            print("프로그램을 종료합니다.")
            return
        
        else:
            print("입력 오류: 0, 1, 2 중 하나를 선택하세요.")


if __name__ == "__main__":
    main()