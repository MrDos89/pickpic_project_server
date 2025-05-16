package com.febrie.util;

import java.text.SimpleDateFormat;
import java.util.Date;

public class Logger {
    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_YELLOW = "\u001B[33m";
    public static final String ANSI_BLUE = "\u001B[34m";
    public static final String ANSI_PURPLE = "\u001B[35m";
    public static final String ANSI_CYAN = "\u001B[36m";
    public static final String ANSI_RED = "\u001B[31m";

    public static void request(String message) {
        log(LogType.REQUEST, message);
    }

    public static void success(String message) {
        log(LogType.SUCCESS, message);
    }

    public static void info(String message) {
        log(LogType.INFO, message);
    }

    public static void warning(String message) {
        log(LogType.WARNING, message);
    }

    public static void error(String message) {
        log(LogType.ERROR, message);
    }
    
    public static void debug(String message) {
        log(LogType.DEBUG, message);
    }

    private static void log(LogType type, String message) {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
        String timestamp = sdf.format(new Date());
        System.out.println(type.getColor() + "[" + timestamp + "] " + type.getPrefix() + ": " + message + ANSI_RESET);
    }

    enum LogType {
        REQUEST(ANSI_BLUE, "요청"),
        SUCCESS(ANSI_GREEN, "성공"),
        INFO(ANSI_CYAN, "정보"),
        WARNING(ANSI_YELLOW, "경고"),
        ERROR(ANSI_RED, "오류"),
        DEBUG(ANSI_PURPLE, "디버그");

        private final String color;
        private final String prefix;

        LogType(String color, String prefix) {
            this.color = color;
            this.prefix = prefix;
        }

        public String getColor() {
            return color;
        }

        public String getPrefix() {
            return prefix;
        }
    }
}
