# DeepChem 2.4.0 공식 이미지 기반
FROM deepchemio/deepchem:2.4.0

# 작업 디렉토리 생성 및 이동
WORKDIR /root/mydir

# 전체 프로젝트 복사
COPY . .

CMD ["/bin/bash"]