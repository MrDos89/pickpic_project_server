package com.febrie.api;

import com.google.gson.Gson;
import org.jetbrains.annotations.NotNull;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

public class ImageSearchApi {

    private static final String SERVER_URL = "http://localhost:8000";

    public static @NotNull String searchByText(String query) throws Exception {
        // 요청 URL 설정
        URL url = new URL(SERVER_URL + "/api/v1/search");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();

        // 요청 설정
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setRequestProperty("Accept", "application/json");
        conn.setDoOutput(true);

        // 요청 바디 설정
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("text", query);

        // JSON 직렬화
        String jsonBody = new Gson().toJson(requestBody);

        // 요청 전송
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = jsonBody.getBytes(StandardCharsets.UTF_8);
            os.write(input, 0, input.length);
        }

        // 응답 받기
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
            return br.lines()
                    .map(String::trim)
                    .collect(Collectors.joining());

        } finally {
            conn.disconnect();
        }
    }
}