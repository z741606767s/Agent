import cv2
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit_drawable_canvas as st_canvas
import os
import time

# 设置TensorFlow日志级别以减少警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

st.set_page_config(page_title="手写数字识别", page_icon="🔢", layout="wide")
st.title("🔢 在线手写数字识别")

# 检查模型文件是否存在
model_path = f"model/mnist.h5"
model = None

if os.path.exists(model_path):
    try:
        # 显示加载进度
        with st.spinner('正在加载模型...'):
            # 尝试加载模型
            model = tf.keras.models.load_model(model_path, compile=False)
            # 重新编译模型
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        st.success("✅ 模型加载成功!")
    except Exception as e:
        st.error(f"❌ 加载模型时出错: {e}")
        model = None
else:
    st.warning("⚠️ 模型文件 'mnist.h5' 不存在，请先运行 train.py 训练模型")
    st.info("ℹ️ 您仍然可以使用画布，但无法进行预测")

# 添加模式选择
mode = st.radio("选择识别模式:", ("单数字识别", "多数字识别"), horizontal=True)

# 添加使用说明
with st.expander("使用说明"):
    if mode == "单数字识别":
        st.markdown("""
        - 在画布上绘制一个数字
        - 系统会自动识别并显示结果
        - 确保数字清晰且居中
        """)
    else:
        st.markdown("""
        - 在画布上绘制多个数字
        - 系统会自动分割并识别每个数字
        - 数字之间请保持一定间距
        - 结果将按从左到右的顺序显示
        """)


def preprocess_image(img):
    # 转换为灰度图
    img = img.convert('L')
    # 调整大小为28x28
    img = img.resize((28, 28))
    img_array = np.array(img)

    # 确保图像是黑底白字（与MNIST一致）
    if np.mean(img_array) > 127:  # 如果平均像素值较高，说明是白底
        img_array = 255 - img_array

    # 归一化
    img_array = img_array.astype('float32') / 255.0

    # 添加批次和通道维度
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def extract_digits(image):
    """从图像中提取单个数字"""
    # 转换为灰度图
    if image.shape[2] == 4:  # 如果有alpha通道
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 应用阈值获取二值图像
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 形态学操作 - 闭运算，连接数字的断开部分
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_rects = []
    for contour in contours:
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 过滤太小的区域（噪声）
        if w > 15 and h > 15 and w * h > 100:  # 增加最小面积要求
            digit_rects.append((x, y, w, h))

    # 如果没有找到数字，尝试使用自适应阈值
    if not digit_rects:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 15 and h > 15 and w * h > 100:
                digit_rects.append((x, y, w, h))

    # 按x坐标排序（从左到右）
    digit_rects.sort(key=lambda rect: rect[0])

    digits = []
    for x, y, w, h in digit_rects:
        # 提取数字区域
        digit_region = gray[y:y + h, x:x + w]

        # 创建黑色背景（与MNIST一致）
        bg = np.zeros((28, 28), dtype=np.uint8)

        # 计算缩放比例，保持宽高比
        scale = min(20 / w, 20 / h)  # 留一些边距
        new_w, new_h = int(w * scale), int(h * scale)

        # 调整数字大小
        resized_digit = cv2.resize(digit_region, (new_w, new_h))

        # 计算放置位置（居中）
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2

        # 将数字放在背景中央
        bg[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_digit

        # 确保数字是白色，背景是黑色
        if np.mean(bg) > 127:
            bg = 255 - bg

        digits.append(bg)

    return digits


# 创建两列布局
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("绘制区域")
    # 创建画布
    canvas = st_canvas.st_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",  # 白色笔画
        background_color="#000000",  # 黑色背景
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
    )

with col2:
    st.subheader("识别结果")

    if canvas.image_data is not None and np.any(canvas.image_data[..., :3] > 10):
        if model is None:
            st.warning("无法进行预测，模型未加载")
            st.image(canvas.image_data, caption="您的绘图", use_column_width=True)
        else:
            if mode == "单数字识别":
                # 将画布数据转换为PIL图像
                pil_image = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA")

                # 显示原始图像
                st.image(pil_image, caption="原始图像", width=150)

                # 预处理图像
                with st.spinner("正在处理图像..."):
                    processed_image = preprocess_image(pil_image)

                # 显示处理后的图像
                st.image(processed_image[0, :, :, 0], caption="处理后的图像", width=100)

                # 进行预测
                with st.spinner("正在识别..."):
                    start_time = time.time()
                    predictions = model.predict(processed_image, verbose=0)
                    inference_time = time.time() - start_time

                predicted_label = np.argmax(predictions[0])
                confidence = predictions[0][predicted_label]

                st.success(f"预测结果: **{predicted_label}**")
                st.write(f"置信度: {confidence:.2%}")
                st.write(f"识别耗时: {inference_time:.3f}秒")

                # 显示所有数字的概率分布
                st.subheader("概率分布")
                probs = predictions[0]
                for i in range(10):
                    col_a, col_b = st.columns([1, 4])
                    with col_a:
                        st.write(f"数字 {i}:")
                    with col_b:
                        st.progress(float(probs[i]), text=f"{probs[i]:.2%}")

            else:  # 多数字识别
                # 提取数字
                with st.spinner("正在分割数字..."):
                    digits = extract_digits(canvas.image_data.astype("uint8"))

                if not digits:
                    st.warning("未检测到数字，请绘制更清晰的数字")
                    st.info("提示: 数字之间保持一定间距，确保笔画清晰")
                else:
                    st.write(f"检测到 {len(digits)} 个数字")

                    results = []
                    confidence_scores = []

                    # 创建多列显示每个数字
                    cols = st.columns(len(digits))

                    for i, digit_img in enumerate(digits):
                        with cols[i]:
                            st.image(digit_img, caption=f"数字 {i + 1}", width=80)

                            # 预处理并预测
                            pil_digit = Image.fromarray(digit_img)
                            processed_digit = preprocess_image(pil_digit)

                            prediction = model.predict(processed_digit, verbose=0)
                            predicted_label = np.argmax(prediction[0])
                            confidence = prediction[0][predicted_label]

                            results.append(str(predicted_label))
                            confidence_scores.append(confidence)

                            st.write(f"预测: **{predicted_label}**")
                            st.write(f"置信度: {confidence:.2%}")

                    # 显示完整结果
                    st.subheader("完整识别结果")
                    result_text = "".join(results)
                    st.success(f"**{result_text}**")

                    # 显示平均置信度
                    avg_confidence = np.mean(confidence_scores)
                    st.write(f"平均置信度: {avg_confidence:.2%}")
    else:
        st.info("请在左侧画布上绘制数字")

# 底部按钮
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("🔄 清除画布", use_container_width=True):
        st.rerun()
with col_btn2:
    if st.button("💾 保存结果", use_container_width=True) and canvas.image_data is not None:
        # 保存画布图像
        im = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA")
        im.save("drawing.png")
        st.success("图像已保存为 'drawing.png'")

# 添加页脚
st.markdown("---")
st.caption("手写数字识别应用 | 基于MNIST数据集训练的卷积神经网络")