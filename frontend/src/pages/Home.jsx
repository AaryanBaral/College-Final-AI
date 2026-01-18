import React from "react";
import Navbar from "../components/Navbar.jsx";
import "../styles/home.css";

export default function Home() {
  return (
    <div className="home-page">
      <Navbar />
      <div className="home-main">
        <div className="home-card-top">
          <h2>
            Disease classification using fundus images. Login required for
            predictions.
          </h2>
        </div>
        <div className="home-card">
          <div>
            <h2>Cataract</h2>
            <img
              src="https://www.datocms-assets.com/33519/1600108532-cataracts-illustration.jpg"
              className="image"
              alt="cataract"
            />
          </div>
          <div className="description">
            A cataract is a condition in which the normally clear lens of the
            eye becomes cloudy, leading to a gradual decline in vision. It
            commonly occurs as a part of the aging process, but can also result
            from eye injuries, certain diseases such as diabetes, prolonged
            exposure to ultraviolet light, or long-term use of corticosteroid
            medications. In some cases, cataracts may be present at birth.
            People with cataracts often experience blurred or hazy vision,
            difficulty seeing at night, increased sensitivity to light and
            glare, and a fading or yellowing of colors. As the cataract
            progresses, these symptoms worsen, making daily activities such as
            reading or driving difficult. Treatment in the early stages may
            involve using stronger glasses or brighter lighting, but the
            definitive treatment is surgical removal of the cloudy lens, which
            is then replaced with an artificial intraocular lens. Cataract
            surgery is one of the most common and successful procedures
            performed worldwide, and it usually restores clear vision and
            improves quality of life.
          </div>
        </div>
        <div className="home-card">
          <div>
            <h2>Diabetic Retinopathy</h2>
            <img
              src="https://rvaf.com/wp-content/uploads/2025/06/Can-Diabetic-Retinopathy-Be-Reversed.webp"
              alt="diabetic retinopathy"
              className="image"
            />
          </div>
          <div className="description">
            Diabetic retinopathy is a serious eye disease that occurs as a
            complication of diabetes mellitus, affecting the light-sensitive
            tissue at the back of the eye called the retina. Prolonged high
            blood sugar levels damage the small blood vessels in the retina,
            causing them to leak fluid or blood, or leading to the growth of
            abnormal new vessels. In the early stages, known as
            non-proliferative diabetic retinopathy, patients may not notice
            symptoms, but gradual vision changes such as blurriness, dark spots,
            or difficulty seeing colors can develop. As the disease progresses
            to the proliferative stage, new fragile blood vessels form, which
            can bleed into the eye and cause severe vision loss or blindness.
            Diabetic retinopathy is one of the leading causes of preventable
            blindness in adults. Regular eye examinations, strict control of
            blood sugar, blood pressure, and cholesterol levels are essential to
            reduce the risk of developing or worsening the condition. Treatment
            options include laser therapy, intravitreal injections, and
            vitrectomy surgery, which aim to stop or slow the progression of
            retinal damage and preserve vision.
          </div>
        </div>
        <div className="home-card">
          <div>
            <h2>Glaucoma</h2>
            <img
              src="https://www.voeyedr.com/wp-content/uploads/2025/01/glaucoma-graphic-scaled.jpg"
              className="image"
              alt="glaucoma"
            />
          </div>
          <div className="description">
            Glaucoma is a group of eye disorders that damage the optic nerve,
            the structure responsible for transmitting visual information from
            the eye to the brain. This damage is most often caused by an
            increase in intraocular pressure (pressure inside the eye), although
            glaucoma can sometimes occur even with normal eye pressure. The
            condition develops gradually and is often called the “silent thief
            of sight” because it typically causes no pain or noticeable symptoms
            in its early stages. Over time, glaucoma leads to a gradual loss of
            peripheral (side) vision, and if left untreated, it can result in
            complete blindness. There are two main types: open-angle glaucoma,
            the most common form, which progresses slowly, and angle-closure
            glaucoma, which develops suddenly and can cause severe eye pain,
            headache, nausea, and blurred vision—requiring immediate medical
            attention. While vision lost to glaucoma cannot be restored, early
            detection and consistent treatment can help preserve remaining
            sight. Management usually includes eye drops, oral medications,
            laser therapy, or surgical procedures to lower intraocular pressure
            and prevent further optic nerve damage.
          </div>
        </div>
      </div>
    </div>
  );
}
