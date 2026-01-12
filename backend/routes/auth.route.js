import express from "express";
import { signin, signup, signout } from "../controllers/auth.controller.js";
import { verifyUser } from "../middlewares/verifyUser.js";
import { errorThrower } from "../utils/errorthrower.js";

export const authRouter = express.Router();

authRouter.post("/signup", signup);
authRouter.post("/signin", signin);
authRouter.delete("/signout", verifyUser, signout);

authRouter.get("/verifyuser", verifyUser, async (req, res) => {
  try {
    res.status(200).json({
      success: true,
      message: req.user,
    });
  } catch (error) {
    res.status(400).json(errorThrower(error.message));
  }
});
