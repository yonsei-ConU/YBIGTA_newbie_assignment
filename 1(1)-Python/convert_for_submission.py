import sys

PATH_1 = "./1-graph-traversal" 
PATH_2 = "./2-stack-queue-deque"
PATH_3 = "./3-divide-and-conquer-multiplication"
PATH_4 = "./4-trie"
PATH_5 = "./5-segment-tree"

ROOT_PATH = {
    "1260": PATH_1,
    "2164": PATH_2,
    "11866": PATH_2,
    "1629": PATH_3,
    "10830": PATH_3,
    "3080": PATH_4,
    "5670": PATH_4,
    "2243": PATH_5,
    "3653": PATH_5,
    "17408": PATH_5
}

# TODO: 루트 폴더에 submission 폴더 생성하기 (FileNotFoundError)
PATH_SUB = "./submission" 

def integrate_file(n: str) -> None:
    # submission 디렉터리 자동 생성
    import os
    os.makedirs(PATH_SUB, exist_ok=True)

    # 숫자 코드 파일 열기 — encoding만 추가
    with open(f"{ROOT_PATH[n]}/{n}.py", encoding="utf-8") as f:
        lines = f.readlines()
    num_code = "".join(filter(lambda x: "from lib import" not in x, lines))

    # lib 코드도 동일하게
    with open(f"{ROOT_PATH[n]}/lib.py", encoding="utf-8") as f:
        lib_code = f.read()

    # 1629 예외 처리
    if n == "1629":
        integrated_code = num_code
    else:
        integrated_code = lib_code + "\n\n\n" + num_code

    folder_num = ROOT_PATH[n][2]
    with open(f"{PATH_SUB}/{folder_num}_{n}.py", "w", encoding="utf-8") as f:
        f.write(integrated_code)


if __name__ == "__main__":
    # python convert_submission.py {file_id} 
    if len(sys.argv) == 2: # 특정 파일만 실행
        file_id = sys.argv[1]
        integrate_file(file_id)    
    else: # 모든 파일 실행
        for file in ROOT_PATH:
            integrate_file(file)
