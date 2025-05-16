package com.febrie.exception;

public class ServerException extends RuntimeException {
    private final int statusCode;
    
    public ServerException(String message, int statusCode) {
        super(message);
        this.statusCode = statusCode;
    }
    
    public ServerException(String message, int statusCode, Throwable cause) {
        super(message, cause);
        this.statusCode = statusCode;
    }
    
    public int getStatusCode() {
        return statusCode;
    }
}
