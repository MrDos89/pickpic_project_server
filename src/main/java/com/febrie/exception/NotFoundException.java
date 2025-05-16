package com.febrie.exception;

public class NotFoundException extends ServerException {
    public NotFoundException(String message) {
        super(message, 404);
    }
}
