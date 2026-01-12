import express from "express";
import dotenv from "dotenv";
import cookieParser from "cookie-parser";
import cors from "cors";
import { checkConnection } from "./database.js";
import { authRouter } from "./routes/auth.route.js";
import { predictionRouter } from "./routes/prediction.route.js";

const app = express();
dotenv.config();

app.use(express.json());
app.use(cors());
app.use(cookieParser());

// Exposing server
app.listen(process.env.PORT_SERVER, () => {
  console.log("Server listening in port ", process.env.PORT_SERVER);
});
app.on("error", (err) => {
  console.log("Error creating server: ", err.message);
});

// Routes
app.use("/api/auth", authRouter);
app.use("/api/prediction", predictionRouter);

// Checking database connection
await checkConnection();

// global error handler
app.use((error, req, res, next) => {
  const statusCode = error.statusCode || 500;
  const message = error.message || "Internal Server Error";
  return res.status(statusCode).json({
    success: false,
    message,
  });
});
