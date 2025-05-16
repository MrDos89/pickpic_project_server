package com.febrie.util;

import com.sun.net.httpserver.HttpExchange;
import org.jetbrains.annotations.NotNull;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.stream.Collectors;

public class HttpUtils {
    
    public static String getClientIp(@NotNull HttpExchange exchange) {
        return exchange.getRemoteAddress().getAddress().getHostAddress();
    }
    
    public static @NotNull String extractKeyFromPath(@NotNull String path, String basePath) {
        if (path.equals(basePath) || path.equals(basePath + "/")) {
            return "";
        }
        
        String key = path.substring(basePath.length());
        if (key.startsWith("/")) {
            key = key.substring(1);
        }
        
        return key;
    }
    
    public static String readRequestBody(HttpExchange exchange) throws IOException {
        try (InputStream inputStream = exchange.getRequestBody();
             BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            return reader.lines().collect(Collectors.joining(System.lineSeparator()));
        }
    }
    
    public static void sendResponse(HttpExchange exchange, int statusCode, String response) throws IOException {
        byte[] responseBytes = response.getBytes(StandardCharsets.UTF_8);
        
        exchange.getResponseHeaders().set("Content-Type", "text/plain; charset=UTF-8");
        exchange.sendResponseHeaders(statusCode, responseBytes.length);
        
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(responseBytes);
        }
    }
}
