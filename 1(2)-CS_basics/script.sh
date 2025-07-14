
# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
if ! command -v conda &> /dev/null; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o ~/miniconda.sh
  bash ~/miniconda.sh -b -p "$HOME/miniconda"
  rm ~/miniconda.sh
  
  # 설치한 conda 초기화 (bash용)
  source "$HOME/miniconda/etc/profile.d/conda.sh"
else
  echo "[INFO] conda 이미 설치되어 있음"
fi
conda init bash
source ~/.bashrc

# Conda 환셩 생성 및 활성화
conda create --name myenv python=3.12 --yes
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1
fi

# 필요한 패키지 설치
conda install mypy -y --quiet

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    base=$(basename "$file" .py)
    echo "[INFO] 실행 중: $file"
    python "$file" < "../input/${base}_input" > "../output/${base}_output"

done

# mypy 테스트 실행 및 mypy_log.txt 저장
echo "[INFO] mypy 테스트 실행"
mypy . > ../mypy_log.txt

# conda.yml 파일 생성
conda env export --name myenv > ../conda.yml

# 가상환경 비활성화
conda deactivate