package com.febrie.server;

import com.febrie.handler.ApiHandler;
import com.febrie.handler.FileHandler;
import com.febrie.handler.HealthCheckHandler;
import com.febrie.util.Config;
import com.febrie.util.Logger;
import com.sun.net.httpserver.HttpServer;
import lombok.Getter;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class ServerManager {
    @Getter
    private final HttpServer server;
    private final AtomicBoolean running = new AtomicBoolean(true);
    private static ServerManager instance;

    private ServerManager() throws IOException {
        server = HttpServer.create(new InetSocketAddress(Config.getPort()), 0);

        server.setExecutor(Executors.newFixedThreadPool(Config.getThreadPoolSize()));

        server.createContext("/file", new FileHandler());
        server.createContext("/api", new ApiHandler());
        server.createContext("/health", new HealthCheckHandler());

        Logger.info("서버가 포트 " + Config.getPort() + "에 생성되었습니다");
    }

    public static synchronized ServerManager getInstance() throws IOException {
        if (instance == null) {
            instance = new ServerManager();
        }
        return instance;
    }

    public void start() {
        server.start();
        running.set(true);
    }

    public void stop() {
        running.set(false);
        server.stop(0);
    }

    public boolean isRunning() {
        return running.get();
    }
}
