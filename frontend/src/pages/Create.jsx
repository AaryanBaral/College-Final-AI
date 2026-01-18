import React from "react";
import Navbar from "../components/Navbar.jsx";
import "../styles/create.css";
import { useRef } from "react";
import { useSelector, useDispatch } from "react-redux";
import { useNavigate } from "react-router";
import { toast } from "react-toastify";
import { deleteUserSuccess } from "../redux/user/userSlice.js";

export default function Create() {
  const reduxData = useSelector((state) => state.user);
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [imageUrl, setImageUrl] = React.useState(null);
  const [prediction, setPrediction] = React.useState(null);
  const [reason, setReason] = React.useState(null);
  const [isUploading, setIsUploading] = React.useState(false);
  const [isPredicting, setIsPredicting] = React.useState(false);

  React.useEffect(() => {
    const checkAuthorization = async () => {
      const response = await fetch("/api/auth/verifyuser", {
        method: "GET",
        credentials: "include",
      });
      const result = await response.json();
      if (!reduxData.currentUser || !result.success) {
        dispatch(deleteUserSuccess());
        navigate("/signin");
        return;
      }
      return;
    };
    checkAuthorization();
  }, []);

  const uploadRef = useRef(null);

  const handleImageUpload = async (e) => {
    e.preventDefault();
    setIsUploading(true);
    try {
      const file = e.target.files[0];
      if (!file || !file.type.startsWith("image/")) {
        toast.error("Invalid file type.");
        setIsUploading(false);
        return;
      }
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch("/api/prediction/uploadImage", {
        method: "POST",
        body: formData,
        credentials: "include",
      });
      const result = await response.json();
      if (result.success == false) {
        toast.error(result.message);
        setImageUrl(null);
        setIsUploading(false);
        return;
      }
      toast.success("Image added successfully.");
      setImageUrl(result.message);
      setIsUploading(false);
      return;
    } catch (error) {
      toast.error(error.message);
      setImageUrl(null);
      setIsUploading(false);
      return;
    }
  };

  // Make the disease prediction
  const handlePrediction = async (e) => {
    e.preventDefault();
    setIsPredicting(true);
    try {
      if (!imageUrl) {
        toast.error("Upload an image to make prediction.");
        setIsPredicting(false);
        return;
      }
      const response = await fetch("/api/prediction/predict", {
        method: "POST",
        headers: {
          "Content-type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({
          imageUrl,
          userid: reduxData.currentUser.id,
        }),
      });
      const result = await response.json();
      if (result.success == false) {
        toast.error(`Failed to make prediction: ${result.message}`);
        setIsPredicting(false);
        setPrediction(null);
        setReason(null);
        return;
      }
      setPrediction(result.message.disease);
      setReason(result.message.reason);
      setIsPredicting(false);
      toast.success("Prediction obtained.");
      return;
    } catch (error) {
      toast.error(`Failed to make prediction: ${error.message}`);
      setIsPredicting(false);
      setPrediction(null);
      setReason(null);
      return;
    }
  };

  return (
    <div className="create-page">
      <Navbar />
      <div className="create-main">
        <div className="create-card">
          <div className="create">
            <h2 className="fundus-image-analysis">Fundus image analysis</h2>
            <div className="image-card">
              <div className="image-container">
                {!imageUrl ? (
                  <div>No image Selected</div>
                ) : (
                  <img src={imageUrl} className="image" />
                )}
              </div>
            </div>
            <div className="create-buttons">
              <button
                className="upload-image-button"
                onClick={() => uploadRef.current.click()}
                disabled={isUploading || isPredicting}
              >
                Choose an image
              </button>
              <button
                className="start-detection-button"
                onClick={handlePrediction}
                disabled={isUploading || isPredicting}
              >
                Start Detection
              </button>
              <input
                type="file"
                accept="image/*"
                className="image-upload-field"
                ref={uploadRef}
                onChange={handleImageUpload}
              />
            </div>
          </div>
          <div className="detection-card">
            <h2 className="detection-result">Detection Result</h2>
            <div className="detection" style={{ padding: '0 20px' }}>
              {!prediction && !reason ? (
                <div>
                  {!isPredicting
                    ? "Prediction results will be shown here."
                    : "Loading..."}
                </div>
              ) : (
                <div>
                  <h3>Disease: {prediction}</h3>
                  <div className="reason-text">
                    {reason &&
                      reason.split("<br>").map((line, i) => {
                        const [title, ...rest] = line.split(":");
                        return (
                          <div key={i}>
                            <strong>{title.trim()}:</strong>
                            {rest.length > 0 && ` ${rest.join(":").trim()}`}
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}
            </div>
            <div className="caution">
              *Seek medical professional before making independent decisions.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}