package com.febrie.util;

import com.febrie.data.DataManager;
import com.febrie.server.Stats;

public class ServerCommands {

    public static void printHelp() {
        Logger.info("===== 서버 명령어 도움말 =====");
        Logger.info("data  - 현재 저장된 모든 데이터 조회");
        Logger.info("stats - 서버 상태 정보 표시");
        Logger.info("help  - 이 도움말 표시");
        Logger.info("test  - 모델 API 테스트");
        Logger.info("exit  - 서버 종료");
    }

    public static void printCurrentData() {
        Logger.info("===== 현재 저장된 데이터 =====");
        int amount = DataManager.getFolders();
        if (amount == 0) {
            Logger.info("저장된 데이터가 없습니다.");
        } else {
            Logger.info("총 " + amount + "폴더, " + DataManager.getFiles() + "개의 항목");
        }
    }

    public static void printServerStats() {
        Logger.info("===== 서버 상태 정보 =====");
        Logger.info("포트: " + Config.getPort());
        Logger.info("처리된 요청 수: " + Stats.getRequestCount());
        Logger.info("데이터 항목 수: " + DataManager.getFolders());
        Logger.info("실행 중인 스레드 수: " + Thread.activeCount());

        long usedMemory = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024);
        long maxMemory = Runtime.getRuntime().maxMemory() / (1024 * 1024);

        Logger.info("JVM 메모리 사용량: " + usedMemory + "MB / " + maxMemory + "MB");
        Logger.info("프로세서 수: " + Runtime.getRuntime().availableProcessors());
    }
}
