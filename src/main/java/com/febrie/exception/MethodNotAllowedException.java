package com.febrie.exception;

public class MethodNotAllowedException extends ServerException {
    public MethodNotAllowedException(String message) {
        super(message, 405);
    }
}
