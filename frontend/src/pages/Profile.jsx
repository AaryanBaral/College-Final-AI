import React from "react";
import Navbar from "../components/Navbar.jsx";
import "../styles/profile.css";
import { useSelector, useDispatch } from "react-redux";
import { useNavigate } from "react-router";
import { toast } from "react-toastify";

export default function Profile() {
  const [pastPredictions, setPastPredictions] = React.useState([]);

  const reduxData = useSelector((state) => state.user);
  const dispatch = useDispatch();
  const navigate = useNavigate();
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

  React.useEffect(() => {
    const getAllPastData = async () => {
      const response = await fetch(
        `/api/prediction/getalluserpredictions/${reduxData.currentUser.id}`,
        {
          method: "GET",
        }
      );
      const result = await response.json();
      if (result.success == false) {
        toast.error(result.message);
        setPastPredictions([]);
        return;
      }
      setPastPredictions(result.message);
      return;
    };
    getAllPastData();
  }, []);

  return (
    <div className="profile-page">
      <Navbar />
      <div className="profile-main">
        <div className="profile-card">
          <div>
            <h2>
              Welcome to your profile, {reduxData?.currentUser?.username}!
            </h2>
            <div className="profile-welcome-text">
              You can see your past uploads below.
            </div>
          </div>
        </div>
        {pastPredictions.length == 0 && (
          <h2 className="loading-text">Loading data...</h2>
        )}
        <div className="user-posts">
          {pastPredictions.map((item, index) => (
            <div key={index} className="card">
              <img src={item.imageurl} alt="fundus image" className="image" />
              <div className="card-content">
                <h3>{item.disease}</h3>
                <div>
                  {/* <strong><h3>Reason:</h3></strong> */}
                  <div className="reason-text">
                    {item.reason.split("<br>").map((line, i) => {
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
                <div><strong>Created:</strong> {new Date(item.created_at).toLocaleString()}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
