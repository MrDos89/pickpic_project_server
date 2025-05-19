package com.febrie.handler;

import com.febrie.api.ModelApi;
import com.febrie.exception.MethodNotAllowedException;
import com.febrie.exception.NotFoundException;
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

        String[] uri = exchange.getRequestURI().toString().split("/");
        String key = uri[2];
        String ssid = uri[3];

        try {
            if (method.equals("POST")) {
                handlePostRequest(exchange, key, ssid, HttpUtils.readRequestBody(exchange));
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
            HttpUtils.sendResponse(exchange, 500, "서버 오류: " + e);
            Logger.error("서버 오류 발생: " + e);
        } finally {
            long elapsedTime = System.currentTimeMillis() - startTime;
            Logger.info("요청 처리 시간: " + elapsedTime + "ms");
        }
    }

    private void handlePostRequest(HttpExchange exchange, @NotNull String key, @NotNull String ssid, String data) throws IOException, NotFoundException {
        JsonObject datajson = JsonParser.parseString(data).getAsJsonObject();
        JsonElement json;
        switch (key) {
            case "pose" -> {
                try {
                    json = JsonParser.parseString(ModelApi.getImageByPoses(ssid,
                            datajson.get("pose").getAsString()));
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
            case "txt2img" -> {
                try {
                    json = JsonParser.parseString(ModelApi.searchByText(ssid,
                            datajson.get("keyword").getAsString()));
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
            case "img2img" -> {
                try {
                    json = JsonParser.parseString(ModelApi.searchByImage(ssid, datajson.get("image_name").getAsString()));
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
            default -> {
                HttpUtils.sendResponse(exchange, 400, "잘못된 요청입니다.");
                Logger.error("잘못된 요청. \n없는 파라미터: " + key);
                return;
            }
        }
        JsonArray result = json.getAsJsonObject().getAsJsonArray("results");
        HttpUtils.sendResponse(exchange, 200, result.toString());
        Logger.success("반환 데이터 개수:" + result.size());
    }
}
