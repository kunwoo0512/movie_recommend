# 코파일럿 성능 개선을 위한 정리 방안

## 즉시 실행 가능한 해결책

### 1. 대용량 파일 이동/정리
```powershell
# 벡터 DB 파일들을 별도 폴더로 이동
mkdir vector_db_backup
move movie_chunk_embeddings.json vector_db_backup/
move movie_embeddings_*.json vector_db_backup/
move faiss_movie_*.* vector_db_backup/
```

### 2. 백업 파일 정리
```powershell
# 백업 파일들 삭제 (18MB+ 절약)
Remove-Item movie_embeddings_bert.json.backup_*
```

### 3. VS Code 재시작
- VS Code 완전 종료 후 재시작
- 워크스페이스 다시 열기

## 장기적 해결책

### 1. .gitignore 설정
```
# 벡터 DB 파일들
*.bin
*embeddings*.json
faiss_movie_*.json
movie_chunk_*.json
vector_db_backup/
```

### 2. VS Code 설정 최적화
```json
{
    "files.exclude": {
        "**/*.bin": true,
        "**/movie_*embeddings*.json": true,
        "**/faiss_movie_*.json": true
    },
    "search.exclude": {
        "**/vector_db_backup": true,
        "**/*.bin": true
    }
}
```

### 3. 메모리 사용량 모니터링
- Task Manager에서 VS Code 프로세스 확인
- 필요시 extension 비활성화

## 현재 상황
- 벡터 DB 파일 총 용량: ~200MB
- VS Code 프로세스 과다 실행 중
- 시스템 여유 메모리: 3.1GB (부족)