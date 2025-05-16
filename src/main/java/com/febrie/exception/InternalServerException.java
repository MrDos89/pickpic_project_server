package com.febrie.exception;

public class InternalServerException extends ServerException {
    public InternalServerException(String message) {
        super(message, 500);
    }
    
    public InternalServerException(String message, Throwable cause) {
        super(message, 500, cause);
    }
}
