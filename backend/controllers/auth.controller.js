import { errorThrower } from "../utils/errorthrower.js";
import { addUserDB, getUserByUsernameDB } from "../database.js";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import dotenv from "dotenv";

dotenv.config();

export const signup = async (req, res) => {
  try {
    const { username, password, confirmPassword } = req.body;

    if (username.trim().length < 5 || username.trim().length > 25) {
      return res
        .status(400)
        .json(errorThrower("Username of length 6-25 required."));
    } else if (password !== confirmPassword) {
      return res.status(400).json(errorThrower("Passwords don't match."));
    }

    const regex =
      /^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,50}$/;
    if (!regex.test(password)) {
      return res
        .status(400)
        .json(
          errorThrower(
            "Password should have atleast one uppercase, lowercase, special character,number, and length of 8-50 characters."
          )
        );
    }
    const hashedPassword = bcrypt.hashSync(password, 10);
    const result = await addUserDB(username.trim(), hashedPassword);
    return res.status(200).json({
      success: true,
      message: result["rows"][0],
    });
  } catch (err) {
    return res.status(400).json(err);
  }
};

export const signin = async (req, res) => {
  try {
    const { username, password } = req.body;
    const result = await getUserByUsernameDB(username);
    const userDetails = result["rows"][0];
    if (!bcrypt.compareSync(password, userDetails.password)) {
      return res.status(400).json(errorThrower("Passwords don't match."));
    }
    const { password: pass, ...rest } = userDetails;
    const token = jwt.sign({ id: userDetails.id }, process.env.JWT_SECRET_KEY);

    return res
      .status(200)
      .cookie("access_token", token, {
        httpOnly: true,
        maxAge: 1000 * 60 * 60 * 24,
      })
      .json({
        success: true,
        message: rest,
      });
  } catch (error) {
    return res.status(400).json(error);
  }
};

export const signout = async (req, res) => {
  try {
    return res.status(200).clearCookie("access_token").json({
      success: true,
      message: "Logout successful.",
      clearedUser: req.user, // just for clarity, can be omitted
    });
  } catch (error) {
    return res.status(400).json(errorThrower(error.message));
  }
};