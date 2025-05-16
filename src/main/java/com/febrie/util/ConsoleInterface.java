package com.febrie.util;

import com.febrie.server.ServerManager;
import org.jetbrains.annotations.NotNull;

import java.util.Scanner;

public class ConsoleInterface {
    private final ServerManager serverManager;
    private final Scanner scanner;

    public ConsoleInterface(ServerManager serverManager) {
        this.serverManager = serverManager;
        this.scanner = new Scanner(System.in);
    }

    public void start() {
        while (serverManager.isRunning()) {
            System.out.print("\n> ");

            if (scanner.hasNextLine()) {
                String command = scanner.nextLine().trim();
                processCommand(command);
            } else {
                break;
            }
        }

        scanner.close();
    }

    private void processCommand(@NotNull String command) {
        switch (command.toLowerCase()) {
            case "help":
                ServerCommands.printHelp();
                break;

            case "exit":
                exitServer();
                break;

            case "data":
                ServerCommands.printCurrentData();
                break;

            case "stats":
                ServerCommands.printServerStats();
                break;

            case "":
                break;

            default:
                Logger.warning("알 수 없는 명령어입니다. 'help'를 입력하여 도움말을 확인하세요.");
                break;
        }
    }

    private void exitServer() {
        Logger.info("서버를 종료합니다...");
        serverManager.stop();
        Logger.success("서버가 종료되었습니다.");
    }
}
