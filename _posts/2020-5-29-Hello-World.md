---
layout: post
title: 정책 경사 (1)
---

대개 신경망 훈련은 트레이닝 데이터에 대한 가능도 (likelihood)를 높이는 방향으로 이루어진다. 고양이라는 제목이 달려있는 픽셀 덩어리가 신경망에 입력되었을 때 신경망이 유도하는 확률 분포에서 고양이에 해당하는 값이 가능도이고, 이 가능도가 늘어나도록 파라미터를 건드리는 것이 소위 신경망 훈련이다. 

그러나 강화학습에서는 가능도를 논하는 것이 부자연스럽다. 강화학습에서 학습 주체는 외부에서 제공되는 정답이 아니라 보상을 통해 학습하기 때문에 가능도를 계산할 정답이 주어지지 않기 때문이다. 학습 주체는 환경 속에서 획득하는 누적 보상이 가장 커지도록 행위하는 방법을 배워야 한다. 가장 단순한 방법은 학습 주체의 어떤 행위가 장기적으로 높은 누적 보상을 가져왔다면 당근을, 그렇지 못했다면 채찍을 주는 것이다. 이런 철학을 공유하는 방법론들을 정책 경사 (Policy Gradient) 알고리즘이라 부른다.

정답이 제공되어 있다고 가정하자. 다시 말해 상태 $$s$$ 에서는 행위 $$a$$ 가 올바른 행위라 인간 전문가가 알려주었다고 가정하자. 그렇다면 이 문제는 지도학습이므로 단순히 로그 가능도 $$\log \pi_{\theta} (a \mid s)$$를 높이는 방향으로 신경망을 훈련시키면 된다. (현실적으로는 틀린 말이지만 넘어가자.) 만약 이 데이터가 시간에 따라 여러 쌍이 모였다면 전부 더해주면 된다.

$$\sum_t  \log \pi_{\theta} (a_t \mid s_t)$$

PyTorch 코드는 대강 아래와 같이 쓸 수 있을 것이다.

{% highlight python %}
probs = pi(states)
loss = torch.log(probs).gather(1, actions)
{% endhighlight %}

그러나 앞서 말했다시피 강화학습에서는 $$s_t$$ 에서 행위 $$a_t$$ 를 하도록 판단한 주체가 인간 전문가가 아니라 학습 주체이므로 위 방법론을 사용할 순 없다. 그러나 이를 크게 수정하지 않고 싶다면 이런 접근을 취해볼 수도 있겠다. $$s_t$$에서 했던 행위 $$a_t$$가 정답은 아니지만, 얼마나 좋고 나쁜 행위인지 정량화할 수 있다면 이걸 일종의 가중치로 제공하고 싶다.

$$\sum_t (\ \ \ ) \log \pi_{\theta}(a_t|s_t)$$

$$(s_t, a_t)$$ 가 얼마나 좋고 나쁜 쌍인지 판단하는 문제를 강화학습에서는 신뢰 할당 (Credit Assignment) 문제라 부른다. 단순하게, $$(s_t, a_t)$$ 이후로 받은 보상을 모두 더해 가중치로 줄 수 있다.

$$\sum_t (\sum_{t'=t} r_{t'}) \log \pi_{\theta}(a_t|s_t)$$

{% highlight python %}
probs = pi(states)
weights = cumulative_sum(rewards)
loss = weights * torch.log(probs).gather(1, actions)
{% endhighlight %}

그러나 누적 보상이 얼마 이상이어야 '좋다'라고 할 수 있는가? '좋고 나쁨'은 상대적이고 잘 정의되는 개념이 아니다. 가령 마지막 시점에는 항상 +100만큼의 보상이 주어지는 풍요로운 환경에 놓여있다면, 가중치를 계산할 때 항상 -100은 빼 주어야 각 행위들의 좋고 나쁨을 조금 더 엄격하게 판단할 수 있을 것이다.

$$\sum_t (\sum_{t'=t} r_{t'} - b) \log \pi_{\theta}(a_t|s_t)$$

{% highlight python %}
probs = pi(states)
weights = cumulative_sum(rewards)
loss = (weights - baseline) * torch.log(probs).gather(1, actions)
{% endhighlight %}
