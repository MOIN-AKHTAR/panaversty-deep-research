from agents.lifecycle import  AgentHooksBase

# Create a custom hook class for our agent
class AgentHooks(AgentHooksBase):
    async def on_start(self, context, agent):
        print(f"🕘 {agent.name} is starting up!")
    
    async def on_llm_start(self, context, agent, system_prompt, input_items):
        print(f"📞 {agent.name} is asking AI for help")
    
    async def on_llm_end(self, context, agent, response):
        print(f"🧠✨ {agent.name} got AI response")
    
    async def on_tool_start(self, context, agent, tool):
        print(f"🔨 {agent.name} is using {tool.name}")
    
    async def on_tool_end(self, context, agent, tool, result):
        print(f"✅ {agent.name} finished using {tool.name}")
    
    async def on_end(self, context, agent, output):
        print(f"🎉 {agent.name} completed the task!")