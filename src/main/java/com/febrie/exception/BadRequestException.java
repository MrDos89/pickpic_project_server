package com.febrie.exception;

public class BadRequestException extends ServerException {
    public BadRequestException(String message) {
        super(message, 400);
    }
}
