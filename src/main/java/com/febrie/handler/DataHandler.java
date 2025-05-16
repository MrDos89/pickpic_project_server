package com.febrie.handler;

import com.febrie.api.ImageSearchApi;
import com.febrie.exception.MethodNotAllowedException;
import com.febrie.exception.NotFoundException;
import com.febrie.server.Stats;
import com.febrie.util.HttpUtils;
import com.febrie.util.Logger;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;

public class DataHandler implements HttpHandler {
    @Override
    public void handle(@NotNull HttpExchange exchange) throws IOException {
        long startTime = System.currentTimeMillis();
        String method = exchange.getRequestMethod();

        Stats.incrementRequestCount();

        String[] uri = exchange.getRequestURI().toString().split("/");
        String key = uri[2];

        try {
            if (method.equals("POST")) {
                handlePostRequest(exchange, key, HttpUtils.readRequestBody(exchange));
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

    private void handlePostRequest(HttpExchange exchange, @NotNull String key, String data) throws IOException, NotFoundException {
        JsonObject datajson = JsonParser.parseString(data).getAsJsonObject();
        JsonElement json;
        switch (key) {
            case "pose" -> {

            }
            case "txt2img" -> {
                try {
                    json = JsonParser.parseString(ImageSearchApi.searchByText(datajson.get("ssid").getAsString(),
                            datajson.get("keyword").getAsString()));
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                JsonArray array = json.getAsJsonObject().getAsJsonArray("results");
                JsonObject results = new JsonObject();
                results.add("results", array);
                HttpUtils.sendResponse(exchange, 200, results.toString());
                Logger.success("데이터 조회 완료 - 값 길이: " + array.size() + "개");
            }
            case "img2img" -> {

            }
            default -> {
                HttpUtils.sendResponse(exchange, 400, "잘못된 요청입니다.");
                Logger.error("잘못된 요청. \n없는 파라미터: " + key);
            }
        }
    }
}
