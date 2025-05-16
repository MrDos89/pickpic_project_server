package com.febrie.client;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Base64;
import java.util.Scanner;

public class SimpleClient {
    
    private static final String BASE_URL = "http://localhost:8080";
    
    public static void main(String[] args) {
        try {
            Scanner scanner = new Scanner(System.in);
            
            System.out.println("===== Base64 이미지 업로드 테스트 클라이언트 =====");
            System.out.println("1. 로컬 이미지 파일 업로드");
            System.out.println("2. 서버 상태 확인");
            System.out.print("선택: ");
            
            String choice = scanner.nextLine().trim();
            
            switch (choice) {
                case "1":
                    uploadImage(scanner);
                    break;
                case "2":
                    checkServerStatus();
                    break;
                default:
                    System.out.println("잘못된 선택입니다.");
                    break;
            }
            
        } catch (Exception e) {
            System.err.println("클라이언트 실행 중 오류 발생: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void uploadImage(Scanner scanner) throws IOException {
        System.out.print("업로드할 이미지 파일 경로: ");
        String filePath = scanner.nextLine().trim();
        
        File file = new File(filePath);
        if (!file.exists() || !file.isFile()) {
            System.err.println("파일이 존재하지 않거나 일반 파일이 아닙니다: " + filePath);
            return;
        }
        
        System.out.print("서버에 저장할 이미지 파일명(확장자 제외): ");
        String fileName = scanner.nextLine().trim();
        
        if (fileName.isEmpty()) {
            System.err.println("파일명이 비어 있습니다.");
            return;
        }
        
        byte[] fileBytes = Files.readAllBytes(file.toPath());
        String base64Data = Base64.getEncoder().encodeToString(fileBytes);
        
        System.out.println("파일을 Base64로 인코딩했습니다. 인코딩된 데이터 길이: " + base64Data.length() + "자");
        
        System.out.println("서버에 업로드 중...");
        String response = sendRequest("POST", "/api/" + fileName, base64Data);
        System.out.println("서버 응답: " + response);
    }
    
    private static void checkServerStatus() throws IOException {
        System.out.println("서버 상태 확인 중...");
        String response = sendRequest("GET", "/health", null);
        System.out.println("서버 상태: \n" + response);
    }
    
    private static String sendRequest(String method, String path, String body) throws IOException {
        URL url = new URL(BASE_URL + path);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod(method);
        
        if (body != null && !body.isEmpty()) {
            connection.setDoOutput(true);
            try (OutputStream os = connection.getOutputStream()) {
                byte[] input = body.getBytes(StandardCharsets.UTF_8);
                os.write(input, 0, input.length);
            }
        }
        
        int responseCode = connection.getResponseCode();
        if (responseCode >= 400) {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getErrorStream(), StandardCharsets.UTF_8))) {
                StringBuilder response = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    response.append(line).append("\n");
                }
                throw new IOException("HTTP 오류 코드: " + responseCode + "\n" + response.toString().trim());
            }
        }
        
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8))) {
            StringBuilder response = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                response.append(line).append("\n");
            }
            return response.toString().trim();
        }
    }
}
