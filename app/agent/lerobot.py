from typing import Any

from pydantic import Field
from app.logger import logger
from app.agent.toolcall import ToolCallAgent
from app.prompt.lerobot import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import ToolCollection
from app.tool.robot_action import RobotAction
from app.tool.action_validator import ActionValidator
from app.schema import  AgentState

class Lerobot(ToolCallAgent):
    """
    A versatile general-purpose agent that uses planning to solve various tasks.

    This agent extends PlanningAgent with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    name: str = "Lerobot"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 2000
    max_steps: int = 3


    # Add general-purpose tools to the tool collection 
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            RobotAction()
        )
    )

    # 这个函数用来确定什么使用执行结束
    # self.state = AgentState.FINISHED
    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        # 使用self.arm_validator来判断是否是一个有效的动作
        arm_validator = ActionValidator()
        result = await arm_validator.execute(
            action=self.step_text,
        )
        if result and result.metadata["operation_status"] == "action_execute_successfully":
            self.state = AgentState.FINISHED
            logger.info(
                f"🎯 Tool completed its mission!"
            )
            return

        if not self._is_special_tool(name):
            return
        else:
            await super()._handle_special_tool(name, result, **kwargs)
