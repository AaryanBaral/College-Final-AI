import express from "express";
import {
  getAllUserPredictions,
  getPrediction,
  uploadImage,
} from "../controllers/prediction.controller.js";

import multer from "multer";
const upload = multer({
  limits: 1024 * 1024 * 10, // 10MB limit
  files: 1, // max 1 file
});

export const predictionRouter = express.Router();

predictionRouter.post("/uploadImage", upload.single("file"), uploadImage);

predictionRouter.post("/predict", getPrediction);

predictionRouter.get(`/getalluserpredictions/:userid`, getAllUserPredictions);
