package com.febrie.handler;

import com.febrie.exception.MethodNotAllowedException;
import com.febrie.exception.NotFoundException;
import com.febrie.server.Stats;
import com.febrie.util.Config;
import com.febrie.util.HttpUtils;
import com.febrie.util.Logger;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.io.IOException;

public class ApiHandler implements HttpHandler {
    @Override
    public void handle(@NotNull HttpExchange exchange) throws IOException {
        long startTime = System.currentTimeMillis();
        String method = exchange.getRequestMethod();

        Stats.incrementRequestCount();

        String[] uri = exchange.getRequestURI().toString().split("/");

        try {
            if (method.equals("GET")) {
                handleGetRequest(exchange, uri[2]);
            } else {
                throw new MethodNotAllowedException("허용되지 않는 메서드입니다");
            }
        } catch (MethodNotAllowedException e) {
            HttpUtils.sendResponse(exchange, e.getStatusCode(), e.getMessage());
            Logger.error("지원하지 않는 메서드 - " + method);
        } catch (NotFoundException e) {
            HttpUtils.sendResponse(exchange, e.getStatusCode(), e.getMessage());
            Logger.warning(e.getMessage());
        } catch (Exception e) {
            HttpUtils.sendResponse(exchange, 500, "서버 오류: " + e.getMessage());
            Logger.error("서버 오류 발생: " + e.getMessage());
        } finally {
            long elapsedTime = System.currentTimeMillis() - startTime;
            Logger.info("요청 처리 시간: " + elapsedTime + "ms");
        }
    }

    private void handleGetRequest(HttpExchange exchange, String key) throws IOException, NotFoundException {
        File[] files = new File(Config.getImageSavePath()).listFiles();
        int value = files == null ? 0 : files.length;;
        HttpUtils.sendResponse(exchange, 200, String.valueOf(value));
        Logger.success("데이터 조회 완료 - 값 길이: " + value + "개");
    }
}
