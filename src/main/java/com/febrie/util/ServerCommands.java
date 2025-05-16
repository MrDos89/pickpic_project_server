package com.febrie.util;

import com.febrie.server.Stats;
import com.febrie.store.DataStore;

public class ServerCommands {
    
    public static void printHelp() {
        Logger.info("===== 서버 명령어 도움말 =====");
        Logger.info("data  - 현재 저장된 모든 데이터 조회");
        Logger.info("stats - 서버 상태 정보 표시");
        Logger.info("help  - 이 도움말 표시");
        Logger.info("exit  - 서버 종료");
    }
    
    public static void printCurrentData() {
        DataStore dataStore = DataStore.getInstance();
        Logger.info("===== 현재 저장된 데이터 =====");
        
        if (dataStore.isEmpty()) {
            Logger.info("저장된 데이터가 없습니다.");
        } else {
            dataStore.forEach((key, value) -> {
                if (value.length() > 50) {
                    Logger.info(key + ": " + value.substring(0, 47) + "...");
                } else {
                    Logger.info(key + ": " + value);
                }
            });
            Logger.info("총 " + dataStore.size() + "개 항목");
        }
    }
    
    public static void printServerStats() {
        Logger.info("===== 서버 상태 정보 =====");
        Logger.info("포트: " + Config.getPort());
        Logger.info("처리된 요청 수: " + Stats.getRequestCount());
        Logger.info("데이터 항목 수: " + DataStore.getInstance().size());
        Logger.info("실행 중인 스레드 수: " + Thread.activeCount());
        
        long usedMemory = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024);
        long maxMemory = Runtime.getRuntime().maxMemory() / (1024 * 1024);
        
        Logger.info("JVM 메모리 사용량: " + usedMemory + "MB / " + maxMemory + "MB");
        Logger.info("프로세서 수: " + Runtime.getRuntime().availableProcessors());
    }
}
