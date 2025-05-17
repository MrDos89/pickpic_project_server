package com.febrie;

import com.febrie.server.ServerManager;
import com.febrie.util.Config;
import com.febrie.util.ConsoleInterface;
import com.febrie.util.Logger;

import java.io.IOException;

public class Application {

    public static void main(String[] args) {
        try {
            ServerManager serverManager = ServerManager.getInstance();
            serverManager.start();

            Logger.info("순수 Java REST API 서버가 포트 " + Config.getPort() + "에서 시작 되었습니다!");
            Logger.info("서버를 종료하려면 'exit' 명령어를 입력하세요.");
            Logger.info("사용 가능한 명령어: 'data' (현재 데이터 보기), 'help' (도움말), 'test' (테스트)");
            Logger.debug("디버그 작동 테스트");
            ConsoleInterface consoleInterface = new ConsoleInterface(serverManager);
            consoleInterface.start();

        } catch (IOException e) {
            Logger.error("서버 시작 중 오류 발생: " + e);
        } catch (Exception e) {
            Logger.error("예상치 못한 오류 발생: " + e);
        }
    }
}
