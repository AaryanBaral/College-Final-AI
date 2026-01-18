import { errorThrower } from "../utils/errorthrower.js";
import jwt from "jsonwebtoken";
import dotenv from "dotenv";

dotenv.config();

export const verifyUser = (req, res, next) => {
  try {
    const token = req.cookies.access_token;
    if (!token) {
      const error = errorThrower("Please login to continue.");
      error.statusCode = 401;
      return next(error);
    }
    const validUser = jwt.verify(token, process.env.JWT_SECRET_KEY);
    if (!validUser) {
      const error_unauthorized = errorThrower("Unauthorized user.");
      error_unauthorized.statusCode = 401;
      return next(error_unauthorized);
    }
    req.user = validUser;
    return next();
  } catch (error) {
    return next(errorThrower("Failed to verify user."));
  }
};
