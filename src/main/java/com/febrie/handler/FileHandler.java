package com.febrie.handler;

import com.febrie.data.DataManager;
import com.febrie.exception.InternalServerException;
import com.febrie.exception.MethodNotAllowedException;
import com.febrie.exception.NotFoundException;
import com.febrie.util.Config;
import com.febrie.util.FileManager;
import com.febrie.util.HttpUtils;
import com.febrie.util.Logger;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.io.IOException;

public class FileHandler implements HttpHandler {
    @Override
    public void handle(@NotNull HttpExchange exchange) throws IOException {
        long startTime = System.currentTimeMillis();
        String method = exchange.getRequestMethod();
        String[] uri = exchange.getRequestURI().toString().split("/");

        try {
            switch (method) {
                case "GET":
                    handleGetRequest(exchange);
                    break;
                case "POST":
                case "PUT":
                    handlePostPutRequest(exchange);
                    break;
                case "DELETE":
                    handleDeleteRequest(exchange, uri[2]);
                    break;

                default:
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

    private void handleGetRequest(HttpExchange exchange) throws IOException, NotFoundException {
        HttpUtils.sendResponse(exchange, 200, "업로드 되어 있는 총 파일 개수" + DataManager.getFiles() + "개");
        Logger.success("데이터 조회 완료 - 파일 개수: " + DataManager.getFiles() + "개");
    }

    private void handlePostPutRequest(HttpExchange exchange) throws IOException {
        String jsonString = HttpUtils.readRequestBody(exchange);
        JsonArray jsons = JsonParser.parseString(jsonString).getAsJsonObject().getAsJsonArray("images");
        Logger.debug(jsons.size() + "개의 이미지 데이터를 전달받았습니다.");
        try {
            for (JsonElement json : jsons)
                FileManager.saveBase64ImageToFile(json.getAsJsonObject().get("image_data").getAsString(),
                        json.getAsJsonObject().get("ssid").getAsString(), json.getAsJsonObject().get("uid").getAsString());
            HttpUtils.sendResponse(exchange, 200, "이미지 저장 완료: " + jsons.size() + "개");
            Logger.success("이미지 파일 저장 완료 - 총 개수: " + jsons.size());
        } catch (Exception e) {
            throw new InternalServerException("이미지 저장 실패: " + e, e.getCause());
        }
    }

    private void handleDeleteRequest(HttpExchange exchange, String ssid) throws IOException, NotFoundException {
        File file = new File(Config.getImageSavePath() + ssid + "/");
        if (file.exists()) {
            if (file.delete()) {
                HttpUtils.sendResponse(exchange, 200, "삭제 완료: " + file.getPath());
                Logger.success("파일 삭제 완료: " + file.getPath());
            } else {
                HttpUtils.sendResponse(exchange, 400, "삭제 실패");
                Logger.success("파일 삭제 실패");
            }
        } else {
            HttpUtils.sendResponse(exchange, 404, "삭제할 파일이 없음");
            Logger.error("삭제할 파일이 없음");
        }
    }
}
