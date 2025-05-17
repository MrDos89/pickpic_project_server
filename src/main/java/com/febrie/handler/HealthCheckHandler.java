package com.febrie.handler;

import com.febrie.data.DataManager;
import com.febrie.exception.MethodNotAllowedException;
import com.febrie.util.Config;
import com.febrie.util.HttpUtils;
import com.febrie.util.Logger;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.OperatingSystemMXBean;

public class HealthCheckHandler implements HttpHandler {
    private final MemoryMXBean memoryMXBean = ManagementFactory.getMemoryMXBean();
    private final OperatingSystemMXBean osMXBean = ManagementFactory.getOperatingSystemMXBean();

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        long startTime = System.currentTimeMillis();
        String clientIp = HttpUtils.getClientIp(exchange);
        String method = exchange.getRequestMethod();

        Logger.request(method + " /health (클라이언트: " + clientIp + ")");

        try {
            if (!method.equals("GET")) {
                throw new MethodNotAllowedException("허용되지 않는 메서드입니다. GET만 사용 가능합니다.");
            }

            String response = buildStatusResponse();

            HttpUtils.sendResponse(exchange, 200, response);

            Logger.success("상태 확인 요청 처리 완료");
        } catch (MethodNotAllowedException e) {
            HttpUtils.sendResponse(exchange, e.getStatusCode(), e.getMessage());
            Logger.error("잘못된 메서드 요청 - " + method + " /health");
        } catch (Exception e) {
            Logger.error("상태 확인 처리 중 오류 발생: " + e.getMessage());
            HttpUtils.sendResponse(exchange, 500, "서버 오류: " + e.getMessage());
        } finally {
            long elapsedTime = System.currentTimeMillis() - startTime;
            Logger.info("요청 처리 시간: " + elapsedTime + "ms");
        }
    }

    private @NotNull String buildStatusResponse() {
        return "상태: 정상\n\n" +
                "Java 버전: " + System.getProperty("java.version") + "\n" +
                "OS: " + System.getProperty("os.name") + " " + System.getProperty("os.version") + "\n" +
                "포트: " + Config.getPort() + "\n" +
                "유저 폴더 수: " + DataManager.getFolders() + "\n" +
                "전체 파일 수: " + DataManager.getFiles() + "\n" +
                "메모리 사용량: " + formatMemoryUsage() + "\n" +
                "프로세서 수: " + Runtime.getRuntime().availableProcessors() + "\n" +
                "시스템 부하: " + String.format("%.2f", osMXBean.getSystemLoadAverage()) + "\n" +
                "서버 시간: " + System.currentTimeMillis() + "\n";
    }

    private @NotNull String formatMemoryUsage() {
        long usedHeap = memoryMXBean.getHeapMemoryUsage().getUsed() / (1024 * 1024);
        long maxHeap = memoryMXBean.getHeapMemoryUsage().getMax() / (1024 * 1024);
        return usedHeap + "MB / " + maxHeap + "MB";
    }
}
