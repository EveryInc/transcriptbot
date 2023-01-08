import React, { Component } from 'react';

class ChatBot extends Component {
  constructor(props) {
    super(props);
    this.state = {
      messages: [],
      input: '',
      persona: 'Socrates',
      style: 'default'
    };
    // Bind sendMessage function to component instance
    this.sendMessage = this.sendMessage.bind(this);
  }

  componentDidMount() {
    // Add initial message from bot to messages list
    this.setState({
      messages: [...this.state.messages, {
        sender: "bot",
        message: "How can I help you today?"
      }]
    });
  }

  updateStyle = (style) => {
    this.setState({
      style: style
    });
  }

  updatePersona = (persona) => {
    this.setState({
      persona: persona
    });
  }


  handleChange = (e) => {
    let inputValue = e.target.value;
    let inputColor = 'white';
    if (inputValue.startsWith('/')) {
      inputColor = 'rgb(255, 154, 236)';
    }
    this.setState({input: inputValue, inputColor: inputColor});
  }

  handleSubmit = (e) => {
    e.preventDefault();

    // Add user message to messages list
    this.setState({
      messages: [...this.state.messages, {
        sender: "user",
        message: this.state.input
      }],
      input: ''
    }, () => {
      // Send message to API
      this.sendMessage();
    });
  }

  handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      this.handleSubmit(e);
    }
  }

  sendMessage() {
    // Build transcript string
    let transcript = '';
    this.state.messages.forEach((message) => {
      transcript += `${message.sender}: ${message.message}\n`;
    });

    // Determine promptType based on last message
    let promptType = "completion";
    if (this.state.messages[this.state.messages.length - 1].message === "/summarize") {
      promptType = "summary";
    }

    var messages = this.state.messages;

    this.setState({input: ''})


    // Send transcript to API
    const response_promise = fetch('/ask-huberman', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        transcript: transcript,
        promptType: promptType,
        messages: messages,
        persona: this.state.persona,
        style: this.state.style
      })
    });
    response_promise
    .then(res => res.text())
    .then(response => {
      try {
        response = JSON.parse(response);
        var message = response.answer;
        var topSource = response.sources[0];
        var position = Math.round(topSource.position * 100);

        var newMessages = [...this.state.messages];
        newMessages.push({
          sender: "bot",
          message: message
        });

        if(topSource) {
          newMessages.push({
            sender: "bot",
            message: "For more information see: " + topSource.title + " (" + position + "%" + " of the way through the episode.)"
          });
        }

        this.setState({
          messages: newMessages
        },() => {
          // Scroll to bottom of chatbot-messages div
          this.messagesEnd.scrollIntoView({ behavior: 'smooth' });
        });  
      } catch (e) {
        console.log(e);
      }
      });
  }

  render() {
    return (
      <div className="chatbot-container">
        <div className="chatbot-messages">
          {this.state.messages.map((message, index) => (
            <div key={index} className={`chatbot-message ${message.sender}`}
            ref={(el) => {
              if (index === this.state.messages.length - 1 && el) {
                this.messagesEnd = el;
                this.messagesEnd.scrollIntoView({ behavior: 'smooth' });
              }
            }}>
              {message.message}
            </div>
          ))}
        </div>
        <form onSubmit={this.handleSubmit}>
          <textarea
            value={this.state.input}
            onChange={this.handleChange}
            onKeyDown={this.handleKeyDown}
            style={{color: this.state.inputColor }}
          />
          <button type="submit">Send</button>
        </form>
      </div>
    );
  }
}

export default ChatBot;