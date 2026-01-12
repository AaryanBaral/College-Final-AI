export const errorThrower = (message)=> {
    const error = new Error();
    error.message = message;
    error.success = false;
    return error;
}