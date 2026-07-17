from pathlib import Path

import tempfile
import cv2
import numpy as np
import streamlit as st

from facenet_models import FacenetModel
from PIL import Image
from core.database import FaceDatabase
from core.whispers import adj_list, connected_comps, whispers
from recognizer import recognize


DB_PATH = Path("data/profiles.pkl")
DETECTION_THRESHOLD = 0.87
WHISPERS_THRESHOLD = 0.2846

st.set_page_config(
    page_title="Face Recognition",
)


def load_database():
    if DB_PATH.exists():
        return FaceDatabase.load(DB_PATH)
    return FaceDatabase()


def draw_matches(image, boxes, names):
    output = image.copy()
    if boxes is None:
        return output
    for box, name in zip(boxes, names):
        x1, y1, x2, y2 = box.astype(int)
        if name == "Unknown":
            color = (255, 0, 0)
        else:
            color = (0, 100, 255)
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        cv2.putText(
            output,
            name,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
    return output


if "database" not in st.session_state:
    st.session_state.database = load_database()

if "model" not in st.session_state:
    st.session_state.model = FacenetModel()


st.title("Face Recognition")
st.write("Upload a photo to recognize a face or add a person.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.asarray(image)
    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )
    if st.button("Recognize Face"):
        try:
            boxes, names = recognize(
                image_array,
                st.session_state.database,
                st.session_state.model
            )
            result = draw_matches(
                image_array,
                boxes,
                names
            )
            st.image(
                result,
                caption="Result",
                use_container_width=True
            )
        except Exception as error:
            st.error(str(error))

    st.subheader("Add this person")
    name = st.text_input("Name")

    if st.button("Add to Database"):
        if name.strip() == "":
            st.warning("Enter a name first.")
        else:
            try:
                st.session_state.database.add_image(
                    name=name.strip(),
                    image=image_array,
                    model=st.session_state.model,
                    detection_threshold=DETECTION_THRESHOLD
                )
                st.session_state.database.save(DB_PATH)
                st.success(
                    f"{name.strip()} was added to the database."
                )
            except ValueError as error:
                st.error(str(error))


st.subheader("People in Database")

if len(st.session_state.database) == 0:
    st.write("No people have been added yet.")
else:
    for name in st.session_state.database.profiles:
        st.write(name)
st.divider()
st.subheader("Sort Photos by Person")

uploaded_photos = st.file_uploader(
    "Upload multiple photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="whispers_upload"
)

if st.button("Sort Photos"):
    if len(uploaded_photos) < 2:
        st.warning("Upload at least two photos.")

    else:
        with st.spinner("Sorting photos..."):
            try:
                with tempfile.TemporaryDirectory() as temp_folder:
                    temp_path = Path(temp_folder)
                    image_paths = []

                    for index, uploaded_photo in enumerate(uploaded_photos):
                        filename = f"{index}_{uploaded_photo.name}"
                        photo_path = temp_path / filename

                        with open(photo_path, "wb") as file:
                            file.write(uploaded_photo.getbuffer())

                        image_paths.append(photo_path)

                    nodes, adjacency = adj_list(
                        image_paths,
                        WHISPERS_THRESHOLD
                    )

                    if len(nodes) == 0:
                        st.warning("No usable faces were found.")

                    else:
                        iterations = 200 * len(nodes)

                        whispers(
                            nodes,
                            adjacency,
                            iterations
                        )

                        groups = connected_comps(
                            adjacency,
                            nodes
                        )

                        st.success(
                            f"Sorted the photos into "
                            f"{len(groups)} group(s)."
                        )

                        for group_number, group in enumerate(
                            groups,
                            start=1
                        ):
                            st.subheader(
                                f"Person Group {group_number}"
                            )

                            columns = st.columns(
                                min(3, len(group))
                            )

                            for index, node in enumerate(group):
                                sorted_image = Image.open(
                                    node.image_path
                                ).convert("RGB")

                                original_name = node.image_path.name.split(
                                    "_",
                                    1
                                )[-1]

                                columns[
                                    index % len(columns)
                                ].image(
                                    sorted_image,
                                    caption=original_name,
                                    use_container_width=True
                                )

            except Exception as error:
                st.error(
                    f"Could not sort the photos: {error}"
                )
