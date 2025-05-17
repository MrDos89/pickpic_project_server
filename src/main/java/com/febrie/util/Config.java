package com.febrie.util;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class Config {
    private static int PORT = 8080;
    private static String IMAGE_SAVE_PATH = "../pickpic_project_server/total_model/data/";
    private static int THREAD_POOL_SIZE = 10;

    private Config() {
    }

    static {
        try {
            loadProperties();
        } catch (Exception e) {
            Logger.warning("설정 파일 로드 중 오류 발생: " + e.getMessage());
            Logger.warning("기본 설정 값이 사용됩니다.");
        }
    }

    private static void loadProperties() throws IOException {
        Properties properties = new Properties();

        try (InputStream inputStream = Config.class.getClassLoader().getResourceAsStream("application.properties")) {
            if (inputStream != null) {
                properties.load(inputStream);
                PORT = Integer.parseInt(properties.getProperty("server.port", String.valueOf(PORT)));
                IMAGE_SAVE_PATH = properties.getProperty("image.save.path", IMAGE_SAVE_PATH);
                THREAD_POOL_SIZE = Integer.parseInt(properties.getProperty("thread.pool.size", String.valueOf(THREAD_POOL_SIZE)));

                Logger.info("설정 파일을 성공적으로 로드했습니다.");
            } else {
                Logger.warning("설정 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.");
            }
        }
    }

    public static int getPort() {
        return PORT;
    }

    public static String getImageSavePath() {
        return IMAGE_SAVE_PATH;
    }

    public static int getThreadPoolSize() {
        return THREAD_POOL_SIZE;
    }
}
