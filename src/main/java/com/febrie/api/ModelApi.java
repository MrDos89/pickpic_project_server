package com.febrie.api;

import com.google.gson.Gson;
import org.jetbrains.annotations.NotNull;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

public class ModelApi {
    private static final String SERVER_URL = "http://localhost:8000";

    public static @NotNull String getImageByPoses(String folder, String pose) throws Exception {
        // 요청 URL 설정
        URL url = new URL(SERVER_URL + "/api/v1/pose/detect-pose/" + folder);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();

        // 요청 설정
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setRequestProperty("Accept", "application/json");
        conn.setDoOutput(true);

        // 요청 바디 설정
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("pose_type", pose);
        // JSON 직렬화
        String jsonBody = new Gson().toJson(requestBody);

        // 요청 전송
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = jsonBody.getBytes(StandardCharsets.UTF_8);
            os.write(input, 0, input.length);
        }

        // 응답 코드 확인
        int responseCode = conn.getResponseCode();

        try {
            // 정상 응답인 경우 (2xx)
            if (responseCode >= 200 && responseCode < 300) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    return br.lines()
                            .map(String::trim)
                            .collect(Collectors.joining());
                }
            } else {
                // 오류 응답인 경우 - 오류 내용 읽기
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getErrorStream(), StandardCharsets.UTF_8))) {
                    String errorResponse = br.lines()
                            .map(String::trim)
                            .collect(Collectors.joining());

                    // 오류 응답을 포함한 예외 발생
                    throw new IOException("서버 응답 코드: " + responseCode +
                            " URL: " + url +
                            " 응답 내용: " + errorResponse);
                }
            }
        } finally {
            conn.disconnect();
        }
    }

    public static @NotNull String searchByText(String folder, String query) throws Exception {
        // 요청 URL 설정
        URL url = new URL(SERVER_URL + "/api/v1/search/" + folder);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();

        // 요청 설정
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setRequestProperty("Accept", "application/json");
        conn.setDoOutput(true);

        // 요청 바디 설정
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("text", query);
        requestBody.put("similarity_threshold", "0.23");
        requestBody.put("detail", "false");
        // JSON 직렬화
        String jsonBody = new Gson().toJson(requestBody);

        // 요청 전송
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = jsonBody.getBytes(StandardCharsets.UTF_8);
            os.write(input, 0, input.length);
        }

        // 응답 코드 확인
        int responseCode = conn.getResponseCode();

        try {
            // 정상 응답인 경우 (2xx)
            if (responseCode >= 200 && responseCode < 300) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    return br.lines()
                            .map(String::trim)
                            .collect(Collectors.joining());
                }
            } else {
                // 오류 응답인 경우 - 오류 내용 읽기
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getErrorStream(), StandardCharsets.UTF_8))) {
                    String errorResponse = br.lines()
                            .map(String::trim)
                            .collect(Collectors.joining());

                    // 오류 응답을 포함한 예외 발생
                    throw new IOException("서버 응답 코드: " + responseCode +
                            " URL: " + url +
                            " 응답 내용: " + errorResponse);
                }
            }
        } finally {
            conn.disconnect();
        }
    }

    public static @NotNull String searchByImage(String folder, String image_name) throws Exception {
        // 요청 URL 설정
        URL url = new URL(SERVER_URL + "/api/v1/imgtoimg/search/" + folder);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();

        // 요청 설정
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setRequestProperty("Accept", "application/json");
        conn.setDoOutput(true);

        // 요청 바디 설정
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("image_name", image_name);
        // JSON 직렬화
        String jsonBody = new Gson().toJson(requestBody);

        // 요청 전송
        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = jsonBody.getBytes(StandardCharsets.UTF_8);
            os.write(input, 0, input.length);
        }

        // 응답 코드 확인
        int responseCode = conn.getResponseCode();

        try {
            // 정상 응답인 경우 (2xx)
            if (responseCode >= 200 && responseCode < 300) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    return br.lines()
                            .map(String::trim)
                            .collect(Collectors.joining());
                }
            } else {
                // 오류 응답인 경우 - 오류 내용 읽기
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getErrorStream(), StandardCharsets.UTF_8))) {
                    String errorResponse = br.lines()
                            .map(String::trim)
                            .collect(Collectors.joining());

                    // 오류 응답을 포함한 예외 발생
                    throw new IOException("서버 응답 코드: " + responseCode +
                            " URL: " + url +
                            " 응답 내용: " + errorResponse);
                }
            }
        } finally {
            conn.disconnect();
        }
    }
}
