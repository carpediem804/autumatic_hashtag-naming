# -Intensive-training
Automatic hashtag creation and automatic photo name creation for portrait photos
##
# 자세한 설명은 논문에 설명되어 있습니다. 

# 요약 : 
사진에 대한 해시태그가 트랜드인 요즘 그 트랜드에 알맞은 도움을 주기 위하여 자동 해시태그 생성 및 자동 제목 생성 시스템을 만들었다. 
해시태그 종류는 다음과 같이 5가지가 있다. 
1) 성별 (Ex: #남성적, #여성적), 2) 나이 대 (Ex: #10대, #20대, #30대, #40대 등등), 3) 감정 (Ex: #웃음, #슬픔, #놀람, #짜증, #무표정, #화남), 
4) 얼굴 상 (Ex: #강아지상, #고양이상, #공룡상, #조류상, #말상 등등) 5) 첫인상  (Ex: #카리스마 있는, #자신감이 넘치는, #지적인 모습 등등) 
이렇게 총 다섯가지가 있다. 
이 시스템의 목표는 VGGNet을 통하여 알맞은 해시태그를 적어주고 사진에 나온 해시태그를 기반으로 KNN 학습을 통해 적절한 제목을 만들어 주는 것이다.

# 제한 알고리즘 frame work
![default](https://user-images.githubusercontent.com/33194900/53875288-8df4d800-4047-11e9-9ed2-4a2bd40f0327.png)

## 사용법
나이, 감정, 성별은 microsoft face api를 사용하였습니다 . 따라서 TaeHong_94.py 코드안에 
headers = {
      'ocp-apim-subscription-key': '', //사용자의 key
      'Content-Type': "application/octet-stream",
      'cache-control': "no-cache",
    }
    를 microsoft face api 회원가입 후 수정하셔서 사용하시면 됩니다. 
얼굴상, 첫인상을 위해 사전 작업한 classfication 파일은 carpediem804@naver.com으로 연락주시면 보내드리겠습니다.
제목을 달기위한 데이터는 xxx.xlsx에 있습니다.

# 메인페이지
![default](https://user-images.githubusercontent.com/33194900/53875395-bed50d00-4047-11e9-9969-ce88c4ddb35a.JPG)
![2](https://user-images.githubusercontent.com/33194900/53875424-cdbbbf80-4047-11e9-87c0-60fd1a15658d.JPG)
![3](https://user-images.githubusercontent.com/33194900/53875430-d01e1980-4047-11e9-8ac1-b1042c04dd07.JPG)
# 시작
![default](https://user-images.githubusercontent.com/33194900/53875442-d44a3700-4047-11e9-9185-8cda3d8c519a.JPG)
# 결과
![default](https://user-images.githubusercontent.com/33194900/53875518-fc399a80-4047-11e9-936c-00c8d1d7e7e6.png)
