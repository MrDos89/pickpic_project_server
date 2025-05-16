# 순수 Java REST API 서버

### 예외 클래스
- `ServerException`: 기본 서버 예외
- `NotFoundException`: 404 Not Found 예외
- `BadRequestException`: 400 Bad Request 예외
- `MethodNotAllowedException`: 405 Method Not Allowed 예외
- `InternalServerException`: 500 Internal Server Error 예외

## 엔드포인트

### GET /file
모든 저장된 데이터 조회

### GET /file/{key}
특정 키의 데이터 조회

### POST /file/{key}
Base64로 인코딩된 이미지 저장

### DELETE /api/{key}
특정 키의 데이터 삭제

### GET /health
서버 상태 확인

## 콘솔 명령어
- `data`: 현재 저장된 모든 데이터 조회
- `stats`: 서버 상태 정보 표시
- `help`: 도움말 표시
- `exit`: 서버 종료

## 실행 방법
`Application` 클래스의 `main` 메서드를 실행하여 서버를 시작

## 테스트 방법
POSTMan 같은 프로그램 이용해서 Http 전송으로 테스트 가능

## 개발 환경
- Java 24 Preview
- 외부 라이브러리 의존성 없음 (순수 Java SE API 사용)
